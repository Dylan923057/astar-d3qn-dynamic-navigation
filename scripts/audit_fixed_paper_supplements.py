"""Read-only final evidence checks; never samples environments or trains."""
from __future__ import annotations
from datetime import datetime
import statistics
import torch
import fixed_supplement_common as c
import fixed_qdagger_common as q
import run_fixed_ce_final500 as ce


def audit():
    frozen=ce.verify();q.verify();scenes=c.read(ce.DATA/'frozen500.json')['scenarios']
    counts={};all_rows=[]
    for item in frozen['models']:
        seed=item['seed']
        for method in ('margin','ce'):
            directory=c.ROOT/item['reused_margin'] if method=='margin' else ce.OUT/f'seed_{seed}'
            assert c.read(directory/'completion.json')['complete']
            details=c.rows(directory/'scene_details.csv');c.scene_order(details,scenes)
            failures=c.read(c.ROOT/item['margin_failure_path'] if method=='margin' else directory/'failure_trajectories.json')
            failed=[r for r in details if not c.flag(r['safe_success'])]
            assert {r['scenario_id'] for r in failures}=={r['scenario_id'] for r in failed}
            assert len(failures)==len(failed)
            lookup={r['scenario_id']:r for r in failed}
            for trajectory in failures:
                row=lookup[trajectory['scenario_id']]
                assert len(trajectory['steps'])==int(row['steps'])
                assert trajectory['termination_reason']==row['termination_reason']
                assert trajectory['steps'][-1]['termination_reason']==row['termination_reason']
            expected=c.summarize(details)
            current=next(r for r in c.rows(ce.RESULTS/'per_seed.csv') if r['method']==method and int(r['seed'])==seed)
            for key in ce.KEYS:assert abs(expected[key]-float(current[key]))<1e-10
            counts[f'{method}_seed{seed}']=dict(episodes=500,failures=len(failures));all_rows.append(current)
    for method in ('margin','ce'):
        for key in ce.KEYS:
            values=[float(r[key]) for r in all_rows if r['method']==method]
            row=next(r for r in c.rows(ce.RESULTS/'mean_sample_sd.csv') if r['method']==method and r['metric']==key)
            assert abs(float(row['mean'])-statistics.mean(values))<1e-12
            assert abs(float(row['sample_sd'])-statistics.stdev(values))<1e-12
    qcounts={}
    for seed in (0,1):
        directory=q.OUT/f'seed_{seed}'
        if not (directory/'completion.json').exists():continue
        result=c.read(directory/'result.json')
        assert result['environment_steps']==200000 and result['online_updates']==199501 and result['offline_updates']==20000
        assert result['total_updates']==219501 and result['teacher_queries']==119999
        assert c.sha(directory/'model_final.pth')==result['final_model_sha256']
        curve=c.rows(directory/'validation_curve.csv');details=c.rows(directory/'validation_details.csv')
        assert len(curve)==21 and int(curve[-1]['environment_steps_total'])==200000
        for point in curve:
            selected=[r for r in details if r['environment_steps_total']==point['environment_steps_total']]
            c.scene_order(selected,c.frozen50())
            assert abs(sum(c.flag(r['safe_success']) for r in selected)/50-float(point['safe_success_rate']))<1e-12
        logs=c.rows(directory/'update_diagnostics.csv')
        assert sum(int(r['count']) for r in logs if r['phase']=='offline')==20000
        assert sum(int(r['count']) for r in logs if r['phase']=='online')==199501
        assert all(float(r['weight_max'])==0 for r in logs if r['phase']=='online' and int(r['bin_index'])>=120)
        training=c.rows(directory/'training.csv')
        assert int(training[-1]['environment_steps_total'])==200000 and all(int(r['demo_replay_samples'])==0 for r in training)
        state=torch.load(directory/'recovery_latest.pth',map_location='cpu',weights_only=False)
        assert state['loop']['phase']=='complete' and state['loop']['total']==200000
        assert len(state['loop']['replay'])==10000 and not state['loop']['collector'].pending
        assert state['agent']['update_steps']==219501
        assert all(r.teacher_action==-1 for r in state['loop']['replay'])
        assert state['environment'] is None
        qcounts[str(seed)]=dict(online_steps=200000,offline_updates=20000,online_updates=199501,
            teacher_collection_queries=119999,validation_points=21,validation_rows=len(details),full_recovery_verified=True)
        del state
    report=dict(passed=True,created=datetime.now().isoformat(),ce500_complete=True,ce_counts=counts,
        new_ce_episodes=2500,reused_margin_episodes=2500,qdagger_completed_seeds=qcounts,
        qdagger_all_complete=len(qcounts)==2,policy_rollouts=0,training_updates=0,models_and_old_logs_unchanged=True)
    c.write(report,c.ROOT/'results/asvc_fixed_paper_supplements_v1/evidence_audit.json')
    print('EVIDENCE AUDIT PASS: CE 2500 new + 2500 reused; QDagger completed seeds:',list(qcounts),flush=True)
    return report


if __name__=='__main__':audit()

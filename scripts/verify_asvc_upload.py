"""Read-only, standard-library audit of the ASVC GitHub bundle; never trains or evaluates."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/analysis_upload/asvc_paper_supplement_v1'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    pointer=(BASE/'LATEST.md').read_text(encoding='utf-8')
    match=re.search(r'\]\((\d{8}_\d{6}_\d{6})/README\.md\)',pointer)
    if match is None:raise ValueError('Invalid reading-bundle pointer')
    package=BASE/match.group(1); manifest=read(package/'manifest.json')
    assert manifest['complete'] and manifest['new_training_runs']==5 and manifest['new_evaluation_model_suites']==85
    total=0;maximum=0
    for name,digest in manifest['files_sha256'].items():
        path=(package/name).resolve()
        assert path.is_relative_to(package.resolve()),name
        assert path.is_file() and sha(path)==digest,name
        assert path.suffix.lower() not in ('.pth','.pt','.pkl','.zip','.pem','.key'),name
        size=path.stat().st_size;total+=size;maximum=max(maximum,size)
        assert size<50*1024**2,f'Upload review size threshold exceeded: {name}'
    training=read(package/'registration/training/freeze_manifest.json')
    for name,digest in training['source_sha256'].items():
        assert sha(ROOT/name)==digest,f'Frozen training source changed: {name}'
    assert training['checks']['demonstrations']['transitions']==7260
    assert training['checks']['demonstrations']['observed_risk_waits']==198
    assert training['checks']['recovery']['exact_resume']
    density=read(package/'registration/density/density_registration.json')
    scenes={}
    for name,digest in density['files_sha256'].items():
        path=package/'registration/density'/name
        assert sha(path)==digest and sha(ROOT/'data/asvc_paper_density_v2'/name)==digest
        scenes[str(int(name.split('_')[1].split('.')[0]))]=read(path)['scenes']
    assert len(scenes)==4 and all(len(rows)==500 for rows in scenes.values())
    for i in range(500):
        for n in (1,3,5,7):
            assert scenes[str(n)][i]['obstacles']==scenes['7'][i]['obstacles'][:n]
    for seed in range(5):
        p=package/'run_logs/astar_prefill_bound'/f'seed_{seed}'
        result=read(p/'result.json')
        assert read(p/'completion.json')['complete']
        assert result['environment_steps']==200000 and result['gradient_updates']==199501
        assert result['pretraining_updates']==0 and result['demo_transition_count']==7260
    suites=0;row_count=0
    for path in (package/'evaluation').glob('*/*/seed_*'):
        assert read(path/'completion.json')['complete']
        reg=read(path/'registration.json');summary=read(path/'summary.json')
        assert reg['epsilon']==0 and reg['checkpoint_steps']==200000 and reg['static_mask']
        assert not any(reg[k] for k in ('teacher','route_guidance','dynamic_shield','q_output_clip','used_for_tuning'))
        rows=list(csv.DictReader((path/'scene_details.csv').open(encoding='utf-8',newline='')))
        assert len(rows)==500 and len({r['scenario_id'] for r in rows})==500
        name=path.parent.parent.name
        if name!='original500':
            assert [r['scenario_id'] for r in rows]==[r['scenario_id'] for r in scenes[name]]
            assert reg['scene_sha256']==density['files_sha256'][f'density_{name}.json']
        expected={'safe_success_rate':sum(float(r['safe_success']) for r in rows)/500,
                  'dynamic_collision_rate':sum(float(r['dynamic_collision']) for r in rows)/500,
                  'timeout_rate':sum(r['termination_reason']=='timeout' for r in rows)/500}
        assert all(abs(summary[k]-v)<1e-12 for k,v in expected.items())
        suites+=1;row_count+=len(rows)
    assert suites==85 and row_count==42500
    print(f'PASS: {len(manifest["files_sha256"])} SHA-verified files; five complete 200000-step runs; {suites} evaluation suites / {row_count} scene rows.')
    print(f'Bundle content {total/1024**2:.2f} MiB; largest file {maximum/1024**2:.2f} MiB. Weights and recovery files excluded.')
    print(package.relative_to(ROOT).as_posix())

if __name__=='__main__':verify()

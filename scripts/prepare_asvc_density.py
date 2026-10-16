"""Freeze valid nested density scenes before evaluating them; preserve v1 artifacts."""
from copy import deepcopy
from asvc_supplement_common import *

DENSITY_REG=ROOT/'data/asvc_paper_density_v2'

def prepare():
    config,inputs=configure()
    if (DENSITY_REG/'density_registration.json').exists():
        manifest=common.read_json(DENSITY_REG/'density_registration.json')
        for name,digest in manifest['files_sha256'].items():assert common.sha256(DENSITY_REG/name)==digest
        return
    rng=random.Random(917015108)
    parents=common.read_json(REG/'density_7.json')['scenes']
    result={n:[] for n in (1,3,5,7)};adjustments=0
    for parent in parents:
        parent=deepcopy(parent);occupied=set()
        for spec in parent['obstacles']:
            route=spec['route'];index=spec['start_index']
            if tuple(route[index]) in occupied:
                available=[i for i,p in enumerate(route) if tuple(p) not in occupied]
                if not available:raise ValueError('No distinct starting cell; do not silently resample routes')
                spec['start_index']=rng.choice(available);adjustments+=1
            occupied.add(tuple(route[spec['start_index']]))
        for n in result:
            scene=dict(parent,scenario_id=f'asvc_density_v2_{n}_pair_{parent["pair_id"]:04d}',
                route_ids=parent['route_ids'][:n],categories=parent['categories'][:n],obstacles=parent['obstacles'][:n])
            factory=dq.tail.FrozenFactory(inputs[0],[scene])
            env=factory(inputs[0],max_steps=300,window_size=15,reward_config=legacy._reward_config(config),terminate_on_collision=True)
            env.reset()  # Validates all routes, directions, phases and distinct initial cells.
            result[n].append(scene)
    files={}
    for n,scenes in result.items():
        path=DENSITY_REG/f'density_{n}.json';write_json(dict(scenes=scenes),path);files[path.name]=common.sha256(path)
    manifest=common.read_json(REG/'density_registration.json')
    manifest.update(files_sha256=files,initial_overlap_repair_seed=917015108,initial_phase_adjustments=adjustments,
        all_2000_environments_validated=True,performance_used_for_generation=False,
        validity_rule='Process seven obstacles in frozen order; if initial cell is occupied, uniformly choose among unoccupied cells on the same route. Prefix nesting and routes retained.',
        supersedes='data/asvc_paper_supplement_v1 density scenes: initial overlap validity omission, not a performance-based change',
        representative_scene_ids=[f'asvc_density_v2_5_pair_{i:04d}' for i in (0,1,2)],
        generation_source_sha256=common.sha256(Path(__file__)))
    write_json(manifest,DENSITY_REG/'density_registration.json')
    print(f'Validated and froze 2000 nested scenes; {adjustments} initial overlaps repaired.',flush=True)

if __name__=='__main__':prepare()

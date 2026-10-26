"""Frozen inputs and isolated artifacts for the four-run multitask pilot."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import yaml
from multitask_pilot_domain import GoalFields, scene_for

NAME = 'asvc_multitask_risk_pilot_v1'
CONFIG = ROOT/'configs'/f'{NAME}.yaml'
REG, OUT, RESULTS = (ROOT/folder/NAME for folder in ('data','outputs','results'))
RUNS = (('C',0),('D',0),('C',1),('D',1))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


def immutable(path,value):
    normalized=json.loads(json.dumps(value,ensure_ascii=False))
    if Path(path).exists():
        if read(path)!=normalized:
            raise ValueError(f'Refusing to replace existing input: {path}')
    else:
        write(path,value)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def csv_write(path,rows):
    rows=list(rows)
    if not rows:
        return
    fields=list(dict.fromkeys(key for row in rows for key in row))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def config():
    return yaml.safe_load(CONFIG.read_text(encoding='utf-8'))


def source_paths():
    return [CONFIG,*sorted((ROOT/'scripts').glob('*multitask_pilot*.py')),
            ROOT/'scripts/start_multitask_pilot.ps1',ROOT/'tests/test_multitask_pilot.py',
            *sorted((ROOT/'src').rglob('*.py'))]


def sources():
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in source_paths()}


def load_map(cfg):
    manifest=read(ROOT/cfg['experiment']['map_manifest'])
    data=next(entry['problem'] for entry in manifest['maps'] if entry['problem']['map_id']==cfg['experiment']['map_id'])
    return data,GoalFields(data['size'],data['obstacles'])


def load_data():
    cfg=config()
    data=read(REG/'task_pools.json')
    return cfg,data,GoalFields(data['size'],data['obstacles'])


def training_task(data,cfg,seed,episode):
    # Cycle strata exactly 1:1:1; within every three episodes shuffle their order.
    block=episode//3
    order=list(cfg['tasks']['length_bands'])
    random.Random(cfg['tasks']['sequence_seed_offset']+seed*1000003+block).shuffle(order)
    band=order[episode%3]
    rng=random.Random(cfg['tasks']['sequence_seed_offset']+seed*1000003+episode*1000033)
    choices=[t for t in data['pools']['train'] if t['band']==band]
    return rng.choice(choices),rng.randrange(2**32)


def generate_data(cfg):
    map_data,planner=load_map(cfg)
    settings=cfg['tasks'];rng=random.Random(settings['pool_seed'])
    # Candidate endpoints come only from the original start's connected component.
    original_start=tuple(map_data['start'])
    connected=[tuple(map(int,p)) for p in __import__('numpy').argwhere(planner.fields(original_start)[0]>=0)]
    rng.shuffle(connected)
    goals=connected[:settings['candidate_goals']]
    held_count=round(len(goals)*settings['unseen_goal_fraction'])
    held_goals,seen_goals=goals[:held_count],goals[held_count:]
    starts=[p for p in connected[settings['candidate_goals']:] if p not in held_goals][:settings['candidate_starts']]
    bands=settings['length_bands']
    def candidates(target_goals):
        result={band:[] for band in bands}
        for goal in target_goals:
            distances=planner.fields(goal)[0]
            for start in starts:
                length=int(distances[start])
                for band,(low,high) in bands.items():
                    if low<=length<=high:
                        result[band].append((start,goal,length))
        for bucket in result.values():
            rng.shuffle(bucket)
        return result
    seen=candidates(seen_goals);held=candidates(held_goals)
    used=set()
    def task(pair,band,split,index):
        start,goal,length=pair
        return dict(task_id=f'{split}_{index:03d}',start=start,goal=goal,length=length,band=band,
                    max_steps=max(100,4*length+20),alternative_path=planner.alternative(start,goal))
    def select(buckets,count,split,cover=False):
        quotas={band:count//3+(index<count%3) for index,band in enumerate(bands)}
        chosen=[];uncovered_starts=set(starts);uncovered_goals=set(seen_goals)
        while len(chosen)<count:
            options=[(band,pair) for band,bucket in buckets.items() if quotas[band]>0
                     for pair in bucket if (pair[0],pair[1]) not in used]
            if not options:
                raise ValueError('Not enough statically reachable tasks for disjoint pools')
            if cover and (uncovered_starts or uncovered_goals):
                band,pair=max(options,key=lambda item:int(item[1][0] in uncovered_starts)+int(item[1][1] in uncovered_goals))
            else:
                band,pair=options[0]
            used.add((pair[0],pair[1]));quotas[band]-=1
            uncovered_starts.discard(pair[0]);uncovered_goals.discard(pair[1])
            chosen.append(task(pair,band,split,len(chosen)))
        if cover and (uncovered_starts or uncovered_goals):
            raise ValueError('Training endpoints were not fully covered')
        return chosen
    pools={}
    pools['train']=select(seen,settings['train_count'],'train',True)
    pools['validation']=select(seen,settings['validation_count'],'validation')
    pools['unseen_combination']=select(seen,settings['unseen_combination_count'],'unseen_combination')
    pools['unseen_goal']=select(held,settings['unseen_goal_count'],'unseen_goal')
    data=dict(map_id=map_data['map_id'],size=planner.size,obstacles=sorted(planner.obstacles),
              original_grid_sha256=map_data.get('grid_sha256'),planner_map_digest=planner.map_digest,
              candidate_goals=goals,held_out_goals=held_goals,training_goals=seen_goals,training_starts=starts,
              pools=pools,tie_break=cfg['tasks']['tie_break'])
    validation=[]
    for i,item in enumerate(pools['validation']):
        scene=scene_for(item,planner,cfg['scenes'],settings['pool_seed']+500000+i,(3,4,5)[i%3])
        validation.append(scene)
    previews={str(seed):[dict(episode=i,task_id=training_task(data,cfg,seed,i)[0]['task_id'],
                            scene_seed=training_task(data,cfg,seed,i)[1],obstacle_count=(3,4,5)[i%3])
                       for i in range(300)] for seed in (0,1)}
    return data,validation,previews


def legacy_paths():
    paths=[]
    for folder in ('src','scripts','configs','tests'):
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.yaml','.ps1') and 'multitask' not in path.name:
                paths.append(path)
    current=ROOT/'data/asvc_current_method.json'
    if current.exists():
        paths.append(current)
        # Protect all explicitly registered current model paths without guessing.
        def walk(value):
            if isinstance(value,dict):
                for item in value.values():yield from walk(item)
            elif isinstance(value,list):
                for item in value:yield from walk(item)
            elif isinstance(value,str) and value.endswith('.pth'):
                candidate=Path(value)
                candidate=candidate if candidate.is_absolute() else ROOT/candidate
                if candidate.is_file():yield candidate
        paths.extend(walk(read(current)))
    return sorted(set(paths))


def verify_freeze():
    freeze=read(REG/'frozen_protocol.json')
    if freeze['source_sha256']!=sources():
        raise ValueError('Frozen scientific source/config differs; create a new experiment version')
    for name,digest in freeze['input_sha256'].items():
        if sha(REG/name)!=digest:raise ValueError('Frozen input differs: '+name)
    for name,digest in read(REG/'legacy_protection.json').items():
        if sha(ROOT/name)!=digest:raise ValueError('Protected legacy file changed: '+name)
    return freeze


def run_directory(group,seed):
    if (group,seed) not in RUNS:
        raise ValueError('Only C/D x seed 0/1 pilot runs are authorized')
    return OUT/group/f'seed_{seed}'

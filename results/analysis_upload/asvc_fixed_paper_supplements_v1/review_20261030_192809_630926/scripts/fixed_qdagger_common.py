"""Immutable inputs and audits for the two fixed-task QDagger adapted seeds."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import numpy as np
import fixed_supplement_common as c
from fixed_qdagger_agent import QDaggerAgent, ObservableTeacher, ThreeStepCollector, static_mask
from astar_d3qn.agents.d3qn import D3QNConfig
from astar_d3qn.envs.types import Observation
from astar_d3qn.replay.transition import Transition
from astar_d3qn.utils.seed import seed_everything

ROOT=c.ROOT;NAME='asvc_fixed_qdagger_v1'
DATA,OUT,RESULTS=(ROOT/folder/NAME for folder in ('data','outputs','results'))
CONFIG=ROOT/f'configs/{NAME}.yaml'
CODE=('scripts/fixed_qdagger_agent.py','scripts/fixed_qdagger_common.py','scripts/run_fixed_qdagger.py',
    'scripts/analyze_fixed_qdagger.py','scripts/start_fixed_qdagger.ps1','tests/test_fixed_qdagger.py',
    'scripts/fixed_supplement_common.py',f'configs/{NAME}.yaml','docs/ASVC_FIXED_QDAGGER.zh-CN.md')


def configure():
    config=c.legacy.load_config(CONFIG);shared,inputs=c.configuration()
    assert config['seeds']==[0,1] and config['maximum_new_training_runs']==2
    assert config['environment_steps']==200000 and config['offline_updates']==20000
    assert config['expected_online_updates']==199501 and config['demo_transitions']==7260
    for key,value in dict(learning_rate=.0003,gamma=.99,target_sync_interval=250,gradient_clip_norm=10).items():
        assert shared['agent'][key]==value
    return config,shared,inputs


def make_agent(seed,device=None):
    config,shared,_=configure();seed_everything(seed);values=dict(shared['agent'])
    if device is not None:values['device']=device
    return QDaggerAgent(D3QNConfig(spatial_shape=(5,15,15),scalar_dim=4,action_dim=5,seed=seed,**values))


def baseline_directory(method,seed):
    if method=='margin':return c.ce.baseline_directory(seed)
    names={'d3qn':'whole_map_91701_value_repair_v1/unguided_raw','dqfd':'whole_map_91701_dqfd_comparison_v1/dqfd'}
    return ROOT/'outputs'/names[method]/f'seed_{seed}'


def load_demos():
    config,_,inputs=configure();root=ROOT/'data/whole_map_91701_dqfd_comparison_v1'
    manifest=c.read(root/'freeze_manifest.json')
    for name,expected in manifest['files_sha256'].items():assert c.sha(root/name)==expected
    # NPZ members are lazy: materialize once, never decompress the whole member
    # again for every transition in the identity audit.
    with np.load(root/'demonstrations.npz',allow_pickle=False) as archive:
        data={name:archive[name] for name in archive.files}
    episodes=[e for e in c.read(root/'demonstration_episodes.json')['episodes'] if e['retained']]
    assert manifest['demo_transition_count']==len(data['action'])==7260 and len(episodes)==107
    teacher=ObservableTeacher(inputs[0]);collector=ThreeStepCollector();records=[];audit=[];offset=0
    for ep,episode in enumerate(episodes):
        assert episode['retained_offset']==offset
        for index in range(offset,offset+episode['steps']):
            state=Observation(data['spatial'][index].astype(np.float32),data['scalars'][index])
            next_state=Observation(data['next_spatial'][index].astype(np.float32),data['next_scalars'][index])
            label=teacher.query(state);action=int(data['action'][index])
            assert label==action,f'Demo/online teacher mismatch at {index}'
            assert not state.spatial[4].any() and not next_state.spatial[4].any()
            assert not state.scalars[2:].any() and not next_state.scalars[2:].any()
            end=index==offset+episode['steps']-1
            assert bool(data['episode_end'][index])==end and bool(data['terminated'][index])==end
            if not end:
                assert np.array_equal(data['next_spatial'][index],data['spatial'][index+1])
                assert np.array_equal(data['next_scalars'][index],data['scalars'][index+1])
            assert np.array_equal(static_mask(next_state),data['next_mask'][index])
            transition=Transition(state,action,float(data['reward'][index]),next_state,bool(data['terminated'][index]),data['next_mask'][index])
            records.extend(collector.add(transition,label,ep,episode_end=end))
        assert not collector.pending
        audit.append(dict(episode=ep,scenario_id=episode['scenario_id'],offset=offset,steps=episode['steps'],
            all_teacher_actions_match=True,three_step_never_crosses_episode=True))
        offset+=episode['steps']
    assert len(records)==offset==7260
    return records,dict(original_freeze=manifest,episode_audit=audit,teacher_queries=teacher.queries,
        teacher_seconds=teacher.seconds,teacher_mismatches=0,retained_episodes=107,attempts=128,
        original_demo_generation_seconds=manifest['demonstration_generation_seconds'],
        original_demo_collection_environment_steps=manifest['demo_astar_lookups'],new_demonstration_steps=0,
        n_length_counts={str(k):sum(r.n_length==k for r in records) for k in (1,2,3)})


def verify(check_protected=True):
    frozen=c.read(DATA/'freeze.json');config,shared,_=configure()
    assert frozen['config']==config and frozen['shared_config']==shared
    c.check_files(frozen['source_sha256']);c.check_files(frozen['input_sha256'])
    for name,value in frozen['copied_files_sha256'].items():assert c.sha(DATA/name)==value
    if check_protected:c.check_files(frozen['protected_sha256'])
    return frozen

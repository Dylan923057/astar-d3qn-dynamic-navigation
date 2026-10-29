"""Checks for offline loop evidence; no training and no extra policy rollouts."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import torch
import td_control_common as c
from diagnose_multitask_loops import IndependentFields,annotate_trace,true_teacher_conflict_within4
from diagnose_multitask_pairs import load_agent
from multitask_pilot_domain import make_env,observation_risk
from multitask_pilot_engine import state_digest


class OfflineLoopChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.cfg,cls.data,cls.planner=c.shared();cls.independent=IndependentFields(cls.data)

    def test_independent_teacher_matches_each_task_path(self):
        for task in self.data['pools']['validation']:
            for position in self.planner.path(task['start'],task['goal']):
                self.assertEqual(self.independent.action(task['goal'],position),self.planner.action(task['goal'],position))

    def test_true_future_oracle_restores_environment(self):
        task=self.data['pools']['validation'][0];scene=c.read(c.previous.pilot.REG/'validation_scenes.json')[0]
        env=make_env(task,scene,self.planner,self.cfg);state=env.reset();before=state_digest(env.state_dict())
        first=true_teacher_conflict_within4(env,task['goal'],self.planner)
        self.assertIn(first,(None,1,2,3,4));self.assertEqual(state_digest(env.state_dict()),before)
        # The observation risk API has no environment or true-motion argument.
        with patch.object(env,'_next_dynamic_state',side_effect=AssertionError('hidden future read')):
            risk,_=observation_risk(state,env.position,tuple(task['goal']),self.planner,self.cfg['risk'])
        self.assertGreaterEqual(risk,0);self.assertLessEqual(risk,1)

    def test_saved_loop_annotation_does_not_select_actions_or_update(self):
        base=c.previous.RESULTS/'paired/C/seed_0/static'
        trace=next(r for r in c.read(base/'trajectories.json') if r['task_id']=='validation_012')
        task=next(t for t in self.data['pools']['validation'] if t['task_id']==trace['task_id'])
        scene=next(s for s in c.read(c.previous.pilot.REG/'validation_scenes.json') if s['task_id']==trace['task_id'])
        scene=dict(scene,obstacles=[]);agent=load_agent(c.model_info('C',0),self.cfg)
        before=state_digest(agent.training_state_dict())
        with patch.object(agent,'select_action',side_effect=AssertionError('offline policy re-execution')):
            score,annotations=annotate_trace(agent,task,scene,trace,self.cfg,self.planner,self.independent)
        self.assertEqual(state_digest(agent.training_state_dict()),before)
        self.assertTrue(score['timeout']);self.assertTrue(score['periodic_movement'])
        self.assertEqual(score['goal_input_mismatch'],0);self.assertEqual(score['teacher_label_mismatch'],0)
        self.assertEqual(score['saved_action_not_current_greedy'],0)
        self.assertEqual(len(annotations),score['steps'])
        self.assertTrue(all(r['actual_evaluation_supervision_weight']==0 for r in annotations))
        self.assertTrue(all(r['risk_R']==0 and r['hypothetical_state_C_weight']==r['hypothetical_state_D_weight']==.1 for r in annotations))
        self.assertTrue(all(all(f'q_{a}' in r and f'legal_{a}' in r for a in range(5)) for r in annotations))


if __name__=='__main__':unittest.main(verbosity=2)

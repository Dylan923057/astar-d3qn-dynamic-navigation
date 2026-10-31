"""Namespace/seed-only extension parity and autonomous final500 isolation."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import fixed_qdagger_five_common as f
import run_fixed_qdagger_five as runner


class QDaggerFiveChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_only_seed_path_evaluation_metadata_changes(self):
        new,shared,_=f.configure();old,old_shared,_=f.original.configure()
        self.assertEqual(f.algorithm_config(new),f.algorithm_config(old));self.assertEqual(shared,old_shared)
        self.assertEqual(new['seeds'],[2,3,4]);self.assertEqual(new['environment_steps'],200000)

    def test_original_seeds_cannot_train(self):
        for seed in (0,1):
            with self.assertRaises(ValueError):runner.run_one(seed)

    def test_same_short_training_and_complete_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            for seed in f.ALL_SEEDS:
                before=runner.short_trial(seed,'cpu',Path(temporary)/f'original_{seed}',False)
                after=runner.short_trial(seed,'cpu',Path(temporary)/f'extended_{seed}',True)
                self.assertEqual(before,after,f'Actual offline/online training differs for seed {seed}')

    def test_process_local_binding_restored(self):
        original=runner.original_runner.q;analyze=runner.original_analysis.analyze
        with runner.installed():self.assertIs(runner.original_runner.q,f)
        self.assertIs(runner.original_runner.q,original);self.assertIs(runner.original_analysis.analyze,analyze)

    def test_frozen500_and_evaluation_isolation(self):
        import run_fixed_ce_final500 as ce
        from fixed_qdagger_agent import ObservableTeacher,QDaggerAgent
        _,shared,inputs=f.configure();scenes,digest=f.c.final_scenes();frozen=ce.verify()
        self.assertEqual(digest,frozen['scene_sha256']);self.assertEqual([s['scenario_id'] for s in scenes],frozen['scene_order'])
        agent=f.c.legacy.runtime.make_agent(shared,0,'cpu');before=f.c.digest(agent.training_state_dict())
        with patch.object(ObservableTeacher,'query',side_effect=AssertionError('Teacher used')), \
             patch.object(QDaggerAgent,'train_update',side_effect=AssertionError('Update used')):
            _,details,_=f.c.evaluate(agent,scenes[:1],shared,inputs)
        self.assertEqual(len(details),1);self.assertEqual(before,f.c.digest(agent.training_state_dict()))


if __name__=='__main__':unittest.main()

"""Targeted protocol and backward-compatible training-hook checks."""
import sys
import copy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import run_foundation_dynamic_ratios as experiment
import test_replay_adaptation as fixtures


class FoundationRatioTests(unittest.TestCase):
    def test_time_boundaries_and_integer_batch_counts(self):
        from astar_d3qn.replay.demo import PersistentDemoReplay
        replay = PersistentDemoReplay([object()] * 1320, 8680, 0, 0)
        for step, count in [(1,16),(50000,16),(50001,6),(100000,6),(100001,0),(200000,0)]:
            replay.set_demo_fraction(experiment.fraction('foundation_time_decay', step))
            self.assertEqual(replay.sample_counts(64), (count,64-count))
        self.assertEqual(replay.demonstration_size,1320)
        self.assertEqual(replay.online.capacity,8680)

    def test_test_scenes_independent_unique_and_preserve_validation(self):
        config, legacy, scene_config, problem, entry, pool, frozen = experiment.inputs()
        validation = frozen['scenarios']
        before = experiment.base.state_digest(validation)
        tests = experiment.test_scenes(config,legacy,scene_config,problem,entry,pool,validation)
        signatures = {experiment.scene_signature(s) for s in tests}
        self.assertEqual(len(signatures),200)
        self.assertFalse(signatures & {experiment.scene_signature(s) for s in validation})
        self.assertEqual(before,experiment.base.state_digest(validation))
        self.assertEqual(tests,experiment.test_scenes(config,legacy,scene_config,problem,entry,pool,validation))

    def test_signature_ignores_bookkeeping_but_keeps_motion(self):
        scene = {'obstacles':[{'route':[[1,1],[1,2]],'start_index':0,'direction':1,'move_every':2,'label':'a'}]}
        other = {'obstacles':[{**scene['obstacles'][0],'label':'b'}]}
        self.assertEqual(experiment.scene_signature(scene),experiment.scene_signature(other))
        other['obstacles'][0]['direction'] = -1
        self.assertNotEqual(experiment.scene_signature(scene),experiment.scene_signature(other))

    def test_adaptive_unchanged_rules_and_allows_increases(self):
        c = experiment.TrainingWindowAdaptiveController()
        c.observe_interaction(10000,completed=True,safe_success=True,conflict=True)
        self.assertIsNone(c.ema)
        self.assertEqual(c.rho,.25)
        for step in range(10001,10006):
            c.observe_interaction(step,completed=True,safe_success=True,conflict=True)
        c.observe_interaction(20000,completed=False,safe_success=False,conflict=False)
        self.assertEqual(c.ema,1)
        self.assertEqual(c.rho,0)
        for step in range(20001,20006):
            c.observe_interaction(step,completed=True,safe_success=False,conflict=True)
        c.observe_interaction(30000,completed=False,safe_success=False,conflict=False)
        self.assertAlmostEqual(c.ema,.7)
        self.assertGreater(c.rho,0)

    def test_hooks_do_not_change_constant_ratio_training(self):
        fixture = fixtures.ForkTests()
        fixture.setUp()
        # The old fixture inserts the identical demo objects in online replay;
        # real foundations contain distinct, collected online interactions.
        fixture.base['online_replay']['transitions'] = copy.deepcopy(fixture.base['online_replay']['transitions'])
        stage = dict(max_steps=12,evaluation_interval=4,epsilon_start=.3,epsilon_end=.05,epsilon_decay_steps=15)
        digests = []
        observed = []
        for hooks in (False,True):
            agent,replay = fixtures.restore(fixture.config,fixture.base,.25,42,'cpu')
            kwargs = {} if not hooks else dict(demo_fraction_schedule=lambda step:.25,
                on_batch=lambda step,d,o:observed.append((step,d,o)),
                on_interaction=lambda *args:None)
            result = fixtures.train_steps(agent,replay,fixture.problem,fixture.config,stage,[],42,lambda *a:False,**kwargs)
            digests.append(fixtures.state_digest(fixtures.snapshot(agent,replay)))
        self.assertEqual(digests[0],digests[1])
        self.assertEqual(observed,[(step,2,6) for step in range(1,13)])
        self.assertEqual(result['demo_samples'],24)

    def test_full_evaluation_guard(self):
        fixture = fixtures.ForkTests()
        fixture.setUp()
        agent,replay = fixtures.restore(fixture.config,fixture.base,.25,42,'cpu')
        before = fixtures.state_digest(fixtures.snapshot(agent,replay))
        experiment.preserved_evaluation(agent,replay,fixture.problem,{**fixture.config,'max_episode_steps':12},[None])
        self.assertEqual(before,fixtures.state_digest(fixtures.snapshot(agent,replay)))


if __name__ == '__main__':
    unittest.main()

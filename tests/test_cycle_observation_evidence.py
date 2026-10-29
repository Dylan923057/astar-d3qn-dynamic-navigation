from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from cycle_observation_evidence import functional_cycles


class CycleEvidenceChecks(unittest.TestCase):
    def test_two_state_cycle_detected_once(self):
        result=functional_cycles({'enter':{'a'},'a':{'b'},'b':{'a'}})
        self.assertEqual(len(result),1);self.assertEqual(set(result[0]),{'a','b'})
    def test_wait_is_not_movement_cycle(self):
        self.assertEqual(functional_cycles({'a':{'a'}}),[])
    def test_ambiguous_successor_does_not_prove_closed_cycle(self):
        self.assertEqual(functional_cycles({'a':{'b','c'},'b':{'a'},'c':{'c'}}),[])


if __name__=='__main__':unittest.main(verbosity=2)

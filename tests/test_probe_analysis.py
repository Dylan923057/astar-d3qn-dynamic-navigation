from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_ddqna_stability_probe import probe_aulc


class Checks(unittest.TestCase):
    def test_no_extrapolation_to_200k(self):
        curve=[dict(environment_steps_total=0,safe_success_rate=0),
               dict(environment_steps_total=50000,safe_success_rate=.02)]
        self.assertAlmostEqual(probe_aulc(curve),.01)

    def test_partial_curve_not_a_completed_probe(self):
        with self.assertRaises(ValueError):probe_aulc([dict(environment_steps_total=0,safe_success_rate=0),
                                                     dict(environment_steps_total=40000,safe_success_rate=.5)])


if __name__=='__main__':unittest.main()

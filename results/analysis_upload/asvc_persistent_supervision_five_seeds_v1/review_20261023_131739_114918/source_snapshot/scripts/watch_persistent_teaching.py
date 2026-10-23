"""Finish the authorized study report when all nine training runs complete."""
from __future__ import annotations

import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs/asvc_persistent_teaching_v1'
REPORTS = ROOT/'results/asvc_persistent_teaching_v1'


def main():
    deadline = time.monotonic()+12*3600
    REPORTS.mkdir(parents=True,exist_ok=True)
    while time.monotonic()<deadline:
        rows = []
        for method in ('supervision_floor','advice_floor','both_floors'):
            for seed in (0,1,2):
                directory = OUTPUT/method/f'seed_{seed}'
                status = directory/'status.json'
                try:
                    values = json.loads(status.read_text(encoding='utf-8')) if status.exists() else dict(complete=False,running=False,environment_steps=0)
                except json.JSONDecodeError:
                    continue  # Writer may be replacing the small status document.
                rows.append(dict(method=method,seed=seed,**values))
        (REPORTS/'progress.json').write_text(json.dumps(dict(runs=rows),indent=2),encoding='utf-8')
        failures = [r for r in rows if r.get('error')]
        if failures:
            (REPORTS/'analysis_status.json').write_text(json.dumps(dict(complete=False,training_failures=failures),indent=2),encoding='utf-8')
            raise RuntimeError('Training failed; preserved status and no success conclusion.')
        if len(rows)==9 and all(r['complete'] for r in rows):
            import analyze_persistent_teaching
            output = analyze_persistent_teaching.analyze()
            (REPORTS/'analysis_status.json').write_text(json.dumps(dict(complete=True,report=str(output/'REPORT.md')),indent=2),encoding='utf-8')
            return
        time.sleep(30)
    raise TimeoutError('Training did not complete within 12 hours; inspect processes and progress.')


if __name__=='__main__':main()

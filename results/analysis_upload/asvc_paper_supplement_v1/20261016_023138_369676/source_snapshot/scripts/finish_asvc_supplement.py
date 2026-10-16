"""Wait for authorized existing jobs, then evaluate new models and build the reading bundle. Never trains."""
import time
import traceback
from asvc_supplement_common import *
from evaluate_asvc_supplement import evaluate,EVAL
from package_asvc_supplement import package

def status(stage,**kwargs):
    write_json(dict(stage=stage,complete=False,**kwargs),RESULTS/'pipeline_status.json')
    print(stage,kwargs,flush=True)

def main():
    while True:
        completed=[s for s in SEEDS if (OUT/METHOD/f'seed_{s}/completion.json').exists()]
        errors=[]
        for s in SEEDS:
            p=OUT/METHOD/f'seed_{s}/status.json'
            if p.exists():
                value=common.read_json(p)
                if value.get('error'):errors.append(dict(seed=s,**value))
        if errors:raise RuntimeError(f'Training failed; no automatic retry: {errors}')
        status('waiting_for_five_final_models',completed_seeds=completed)
        if len(completed)==5:break
        time.sleep(30)
    status('evaluating_initialization_control')
    evaluate((METHOD,))
    while True:
        complete=sum((EVAL/str(n)/m/f'seed_{s}'/'completion.json').exists() for n in (1,3,5,7) for m in MAIN for s in SEEDS)
        status('waiting_for_all_density_evaluations',completed=complete,expected=80)
        if complete==80:break
        time.sleep(30)
    status('building_actual_result_figures_and_bundle')
    destination=package()
    write_json(dict(complete=True,stage='complete',reading_bundle=historical.relative(destination)),RESULTS/'pipeline_status.json')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('failed',error=repr(error));traceback.print_exc();raise

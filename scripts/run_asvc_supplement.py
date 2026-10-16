"""Run only the five authorized A* initialization controls; checks precede training."""
import argparse
from asvc_supplement_common import *

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--seeds',nargs='+',type=int,choices=SEEDS,default=list(SEEDS))
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):parser.error('Duplicate seeds')
    preflight()
    if args.train:
        from asvc_prefill_training import run_one
        for seed in args.seeds:run_one(seed,args.resume)

if __name__=='__main__':main()

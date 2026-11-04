"""Evaluate all SERE windows with five patient-held-out models, no selection/refit."""
import argparse
import json
import signal

from stamp.sere.data import read_json
from stamp.sere.checkpoint import run_lock
from stamp.sere_cv.workflow import initialize,prepare,train,test


def stop(signum,frame):
    raise SystemExit(128+signum)


def main():
    for sig in (signal.SIGHUP,signal.SIGTERM):
        signal.signal(sig,stop)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','train','test','all'])
    parser.add_argument('--config',default='configs/sere_full_patient_cv.json')
    parser.add_argument('--reduction',choices=['none','mean','both'],default='both')
    parser.add_argument('--devices',nargs='+',default=['cuda:0'])
    args=parser.parse_args()
    config=read_json(args.config)
    for reduction in (['none','mean'] if args.reduction=='both' else [args.reduction]):
        manifest,folds,root,signature=initialize(config,reduction)
        print(f'Full-trial 5-fold evaluation; {reduction}; output={root}',flush=True)
        with run_lock(root/'orchestrator'):
            if args.phase in ('prepare','all'):
                prepare(config,manifest,root,signature,reduction,args.devices[0])
            if args.phase in ('train','all'):
                train(config,manifest,folds,root,signature,reduction,args.devices)
            if args.phase in ('test','all'):
                report=test(config,manifest,folds,root,signature,reduction,args.devices[0])
                print(json.dumps([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in report['rows']],indent=2))


if __name__=='__main__':
    main()

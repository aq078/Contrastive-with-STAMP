"""Whole-video PennAction: provided split, five seeds, baseline and SupCon."""
import argparse
import json
import signal
from collections import Counter

from stamp.sere.data import read_json
from stamp.sere.checkpoint import run_lock
from stamp.pennaction.data import build_manifest, normalized_video
from stamp.pennaction.workflow import initialize, train, test, validate_config
from stamp.pennaction.embedding import prepare
from pathlib import Path


def stop(signum, frame):
    raise SystemExit(128 + signum)


def main():
    for sig in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(sig, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['audit', 'prepare', 'train', 'test', 'all'])
    parser.add_argument('--config', default='configs/pennaction_video.json')
    parser.add_argument('--reduction', choices=['none', 'mean', 'both'], default='both')
    parser.add_argument('--devices', nargs='+', default=['cuda:0'])
    args = parser.parse_args()
    config = read_json(args.config)
    validate_config(config)
    if args.phase == 'audit':
        manifest = build_manifest(config)
        # Read/validate full training and validation sequences; no embeddings or writes.
        for row in manifest['videos']:
            if row['split'] != 'test':
                normalized_video(Path(config['source_root']) / 'labels' / f"{row['video']}.mat")
        print(json.dumps(dict(counts=Counter(r['split'] for r in manifest['videos']),
            excluded=manifest['excluded'], classes=manifest['classes'],
            max_frames=max(r['frames'] for r in manifest['videos']),
            manifest_hash=manifest['hash']), indent=2))
        return
    for reduction in (['none', 'mean'] if args.reduction == 'both' else [args.reduction]):
        root, manifest, signature = initialize(config, reduction)
        print(f'PennAction whole videos; reduction={reduction}; results={root}', flush=True)
        with run_lock(root / 'orchestrator'):
            if args.phase in ('prepare', 'all'):
                prepare(config, manifest, root, signature, reduction, ['train', 'val'], args.devices[0])
            if args.phase in ('train', 'all'):
                train(config, manifest, root, signature, reduction, args.devices)
            if args.phase in ('test', 'all'):
                result = test(config, manifest, root, signature, reduction, args.devices[0])
                print(json.dumps(result['rows'], indent=2))


if __name__ == '__main__':
    main()

"""Rescore saved CV confusion matrices into a separate report; no model/test access.

This is a diagnostic of OLD unweighted models, never a training selection
artifact for the new weighted experiment.
"""
import argparse
import json
from pathlib import Path

from stamp.sere.data import read_json
from stamp.sere.metrics import CLASS_FIRST

if __package__:
    from .sere_single_seed import aggregate
else:
    from sere_single_seed import aggregate


def rescore(source):
    rows = []
    for mode in ('without_reduction', 'with_reduction'):
        root = source / mode
        validation = read_json(root / 'validation_results.json')
        folds = validation['folds']
        patients = sorted(p for f in folds for p in f['validation_patients'])
        seeds = validation['selection_seeds']
        for method, lengths in validation['methods'].items():
            for length, old in sorted(lengths.items(), key=lambda item: int(item[0])):
                runs = {}
                for fold in folds:
                    for seed in seeds:
                        suffix = Path(f"fold_{fold['fold_id']}") / f'seed_{seed}'
                        cv = root / 'cv' / f'L{length}'
                        path = cv / ('baseline' if method == 'baseline' else 'head_curves') / suffix
                        if method == 'supcon':
                            request = read_json(path / 'request.json') if (path / 'request.json').exists() else {}
                            if request.get('initialization', {}).get('epochs') != old['stage1_epochs']:
                                path = cv / f"head_curves_seed{seed}_stage1epoch{old['stage1_epochs']}" / suffix
                            request = read_json(path / 'request.json')
                            if request['initialization']['epochs'] != old['stage1_epochs']:
                                raise ValueError(f'Incorrect stage 1 initialization: {path}')
                        run = read_json(path / 'run.json')
                        identity = run['identity']
                        if (identity['seed'] != seed or identity['method'] != method
                                or identity['held_out_patients'] != sorted(fold['validation_patients'])):
                            raise ValueError(f'CV identity mismatch: {path}')
                        runs[f"{fold['fold_id']}/{seed}"] = run
                corrected = aggregate(runs, folds, patients, seeds, 'ba', CLASS_FIRST)
                rows.append(dict(reduction=mode, method=method, length=int(length),
                                 old_epoch=old['epochs'], old_score=old['score'],
                                 rescored=corrected))
    return dict(kind='diagnostic_rescore_of_old_unweighted_models', aggregation=CLASS_FIRST,
                source=str(source.resolve()), test_accessed=False, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error('Output must be outside the previous experiment directory')
    report = rescore(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Fail rather than overwrite any existing report.
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    for row in report['rows']:
        new = row['rescored']
        print(f"{row['reduction']} {row['method']} L{row['length']}: "
              f"{new['score']:.4f} +/- {new['fold_std_ba']:.4f}, epoch {new['epoch']}")


if __name__ == '__main__':
    main()

"""Final development fits and untouched tests for every prespecified window."""
import copy
import csv
from pathlib import Path

from stamp.sere.data import file_digest, read_json, write_json
from stamp.sere.checkpoint import resolve_run
from stamp.sere.workflow import (data_path, fit, test, load_locked, lock_artifact)


def window_context(config, output, protocol_hash, validation, length):
    output = Path(output)
    root = output / 'window_evaluation' / f'L{length}'
    snapshot = read_json(output / 'protocol.json')
    path = root / 'protocol.json'
    if path.exists() and read_json(path) != snapshot:
        raise ValueError('Per-window protocol differs from the verified parent')
    write_json(path, snapshot)
    payload = dict(protocol_hash=protocol_hash, reduction=config['reduction'],
                   validation_results_hash=validation['hash'], selection_seeds=config['selection_seeds'],
                   evaluation='fixed_window_no_window_selection', length=length,
                   methods={m: validation['methods'][m][str(length)] for m in ('baseline', 'supcon')})
    # This is a fixed-length epoch plan, not a competition between window lengths.
    lock_artifact(root / 'selection.json', payload)
    window_config = copy.deepcopy(config)
    window_config['window_lengths'] = [length]
    # Reuse the parent development cache through its verified cache descriptor.
    # Legacy caches keep their historical descriptor via prepared_data_dir.
    if data_path(output, length, 'development').exists():
        window_config['prepared_data_dir'] = str(output.parent)
    return window_config, root


def validate_plan(config, output, protocol_hash):
    validation = load_locked(Path(output) / 'validation_results.json', protocol_hash)
    if (validation.get('selection_seeds') != config['selection_seeds']
            or validation.get('window_lengths') != config['window_lengths']
            or validation.get('evaluation') != 'all_candidate_windows'):
        raise ValueError('Run select first to complete the current per-window validation plan')
    return validation


def fit_all_windows(config, manifest, output, protocol_hash, devices):
    validation = validate_plan(config, output, protocol_hash)
    results = {}
    for length in config['window_lengths']:
        window_config, root = window_context(config, output, protocol_hash, validation, length)
        print(f'Final fitting L{length}; seeds={window_config["seeds"]}', flush=True)
        results[str(length)] = fit(window_config, manifest, root, protocol_hash, devices)
    return lock_artifact(Path(output) / 'all_window_final_models.json', dict(
        protocol_hash=protocol_hash, validation_results_hash=validation['hash'], final_seeds=config['seeds'],
        windows={n: r['hash'] for n, r in results.items()}))


def test_all_windows(config, manifest, output, protocol_hash, device):
    validation = validate_plan(config, output, protocol_hash)
    contexts = {}
    # Verify EVERY window/method/seed model before any test samples are prepared.
    for length in config['window_lengths']:
        window_config, root = window_context(config, output, protocol_hash, validation, length)
        final = load_locked(root / 'final_models.json', protocol_hash)
        plan = load_locked(root / 'selection.json', protocol_hash)
        if final['selection_hash'] != plan['hash']:
            raise ValueError('Final models do not match the per-window epoch plan')
        for method in ('baseline', 'supcon'):
            if set(final['runs'][method]) != {str(s) for s in config['seeds']}:
                raise ValueError('Incomplete final seed set; finish fit for all windows first')
            for seed in config['seeds']:
                run = resolve_run(final['runs'][method][str(seed)], root)
                if file_digest(run['checkpoint']) != run['checkpoint_sha256']:
                    raise ValueError('A final checkpoint is missing or changed')
        contexts[length] = (window_config, root)
    rows = []
    for length, (window_config, root) in contexts.items():
        result = test(window_config, manifest, root, protocol_hash, device)
        for method in ('baseline', 'supcon'):
            val = validation['methods'][method][str(length)]
            measured = result['methods'][method]
            row = dict(reduction=config['reduction'], method=method, window_length=length,
                             selection_seed=config['selection_seeds'][0], validation_ba=val['score'],
                             validation_fold_mean_ba=val['fold_mean_ba'],
                             validation_fold_std_ba=val['fold_std_ba'], validation_fold_std_ddof=1,
                             stage1_epochs=val.get('stage1_epochs'), epochs=val['epochs'],
                             test_ba=measured['mean_ba'])
            if 'diagnostics_per_seed' in measured:
                seed = str(config['selection_seeds'][0])
                for prefix, diagnostics in [('validation', val['diagnostics_per_seed'][seed]),
                                            ('test', measured['diagnostics_per_seed'][seed])]:
                    for name in ('negative_recall', 'positive_recall', 'pooled_ba',
                                 'predicted_positive_rate', 'always_positive_ba', 'confusion_matrix'):
                        row[f'{prefix}_{name}'] = diagnostics[name]
                row['metric_aggregation'] = config['metric_aggregation']
            rows.append(row)
    report = lock_artifact(Path(output) / 'evaluation_results.json', dict(
        protocol_hash=protocol_hash, validation_results_hash=validation['hash'],
        final_seeds=config['seeds'], evaluation='all_candidate_windows', rows=rows))
    target = Path(output) / 'evaluation_results.csv'
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(target)
    return report

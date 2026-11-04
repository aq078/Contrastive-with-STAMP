"""Audited candidate restriction, separate from immutable per-run training identity."""
import copy
from pathlib import Path

from stamp.sere.data import read_json, write_json, digest
from stamp.sere.workflow import load_locked, lock_artifact


def final_work_started(output):
    output = Path(output)
    return (any((output / name).exists() for name in (
        'window_evaluation', 'final', 'final_models.json', 'all_window_final_models.json',
        'test', 'test_results.json', 'evaluation_results.json'))
        or any((output / 'data').glob('*/test')))


def replace_with_history(path, previous, payload):
    """Preserve the previous checksummed artifact before atomic replacement."""
    path = Path(path)
    archive = path.parent / 'policy_history' / f"{path.stem}_{previous['hash']}.json"
    if archive.exists() and read_json(archive) != previous:
        raise ValueError('Policy history differs; refusing to overwrite')
    write_json(archive, previous)
    payload = copy.deepcopy(payload)
    payload['hash'] = digest(payload)
    write_json(path, payload)
    return payload


def apply_selection_policy(config, config_path, output, protocol_hash):
    """Call under the orchestrator lock, after normal protocol verification.

The original protocol and training implementation remain immutable. Restricting
which independent fits enter selection changes neither their data nor training.
Record that restriction explicitly instead of rewriting checkpoint signatures.
"""
    config = copy.deepcopy(config)
    sidecar = Path(config_path).with_suffix('.selection.json')
    destination = Path(output) / 'selection_policy.json'
    saved = load_locked(destination, protocol_hash) if destination.exists() else None
    requested = read_json(sidecar) if sidecar.exists() else saved
    if requested is None:
        return config
    lengths = requested['window_lengths']
    original = config['window_lengths']
    approved = set(original) | set(saved['window_lengths'] if saved else [])
    if (not isinstance(lengths, list) or not lengths
            or any(type(n) is not int or n <= 0 for n in lengths)
            or len(set(lengths)) != len(lengths)):
        raise ValueError('Selection window lengths must be distinct positive integers')
    if not set(lengths) <= approved and requested.get('allow_new_window_lengths') is not True:
        raise ValueError('New window lengths require explicit allow_new_window_lengths approval')
    lengths = sorted(lengths)
    payload = dict(protocol_hash=protocol_hash, kind='candidate_window_restriction',
                   original_window_lengths=original, window_lengths=lengths,
                   reason=requested.get('reason', 'User restricted the candidate window lengths'),
                   methods=['baseline', 'supcon'])
    # A later CLI invocation must not silently change an already recorded search.
    expansion = False
    if saved and any(saved.get(k) != value for k, value in payload.items()):
        expansion = (requested.get('allow_window_expansion') is True
                     and set(saved['window_lengths']) < set(lengths)
                     and all(saved.get(k) == v for k, v in payload.items()
                             if k not in ('window_lengths', 'reason')))
        if not expansion:
            raise ValueError('Selection restriction differs from the saved policy; keep the existing policy')
        if final_work_started(output):
            raise ValueError('Cannot expand windows after final fitting or test access')
    if not saved:
        # Restrict selection before final training/testing, not in response to test results.
        if ((Path(output) / 'final').exists() or (Path(output) / 'final_models.json').exists()
                or (Path(output) / 'test').exists() or (Path(output) / 'test_results.json').exists()
                or any((Path(output) / 'data').glob('*/test'))):
            raise ValueError('Cannot introduce a selection restriction after final fitting or test access')
    selection_path = Path(output) / 'selection.json'
    if selection_path.exists():
        selection = load_locked(selection_path, protocol_hash)
        for method in ('baseline', 'supcon'):
            if ({int(n) for n in selection['candidates'][method]} != set(lengths)
                    or selection['methods'][method]['length'] not in lengths):
                raise ValueError('Completed selection used different candidates; refusing to reuse it')
    if expansion:
        replace_with_history(destination, saved, payload)
    elif not saved:
        lock_artifact(destination, payload)
    config['window_lengths'] = lengths
    seed_path = Path(output) / 'selection_seed_policy.json'
    saved_seeds = load_locked(seed_path, protocol_hash) if seed_path.exists() else None
    seeds = requested.get('selection_seeds', saved_seeds['selection_seeds'] if saved_seeds else None)
    if seeds is not None:
        if (not isinstance(seeds, list) or len(seeds) != 1 or type(seeds[0]) is not int
                or seeds[0] not in config['seeds']):
            raise ValueError('Select exactly one of the original training seeds')
        seed_payload = dict(protocol_hash=protocol_hash, selection_seeds=seeds,
                            final_seeds=seeds,
                            original_seeds=config['seeds'], window_lengths=lengths,
                            evaluation='all_candidate_windows',
                            aggregation=config.get('metric_aggregation', 'equal_patients'), variability='sample_std_across_fold_BA_ddof1',
                            reason='User requested one fixed seed for selection and fold-based BA variability')
        if not saved_seeds and ((Path(output) / 'window_evaluation').exists() or (Path(output) / 'final').exists()
                or (Path(output) / 'final_models.json').exists() or (Path(output) / 'test').exists()
                or (Path(output) / 'test_results.json').exists()
                or any((Path(output) / 'data').glob('*/test'))):
            raise ValueError('Cannot change selection seeds after final fitting or test access')
        if saved_seeds and saved_seeds.get('window_lengths') != lengths:
            # Also permits retry after the window policy was committed but the
            # process stopped before this seed-policy update.
            if (requested.get('allow_window_expansion') is not True
                    or not set(saved_seeds['window_lengths']) < set(lengths)
                    or any(saved_seeds.get(k) != v for k, v in seed_payload.items()
                           if k != 'window_lengths') or final_work_started(output)):
                raise ValueError('Cannot change the saved seed policy')
            replace_with_history(seed_path, saved_seeds, seed_payload)
        else:
            lock_artifact(seed_path, seed_payload)
        config['selection_seeds'] = seeds
        config['seeds'] = seeds
    return config

"""Patient-grouped CV, portable epoch resume, and reserved final testing."""
import copy
from pathlib import Path

import torch

from .data import (PROTOCOL, SEEDS, create_manifest, digest, file_digest, iter_windows,
                   patient_folds, read_json, store_index, verify_sources, write_json, write_store)
from .embedding import EmbeddingStore, embed_store, embedding_provenance, load_moment
from .metrics import CLASS_FIRST, aggregate_test, average_fold_curves
from .training import predict_final
from .checkpoint import portable_run, resolve_run, run_lock
from .workers import dispatch


def source_code_hash():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / 'sere').glob('*.py')) + sorted((root / 'modeling').glob('*.py'))
    return digest({str(p.relative_to(root)): file_digest(p) for p in paths})


def reduction_output(config):
    reduction = config.get('reduction', 'none')
    if reduction not in ('none', 'mean'):
        raise ValueError(f'Unknown MOMENT reduction: {reduction}')
    return Path(config['output_dir']).resolve() / ('with_reduction' if reduction == 'mean' else 'without_reduction')


def folds_for(config, manifest):
    cv = config.get('cv', {'n_folds': 5, 'seed': 42})
    return patient_folds(manifest['patients'], cv['n_folds'], cv['seed'])


def initialize(config_path, reduction=None):
    config = read_json(config_path)
    config['reduction'] = reduction if reduction is not None else config.get('reduction', 'none')
    config.setdefault('cv', {'n_folds': 5, 'seed': 42})
    if config['protocol'] != PROTOCOL or config['seeds'] != SEEDS:
        raise ValueError('Protocol and existing five seeds must match')
    lengths = config['window_lengths']
    if not lengths or len(set(lengths)) != len(lengths) or any(type(n) is not int or n < 1 for n in lengths):
        raise ValueError('Window lengths must be distinct positive integers')
    if type(config['stride']) is not int or config['stride'] < 1:
        raise ValueError('Stride must be a positive integer')
    if set(config['recipes']) != {'baseline', 'supcon_stage1', 'supcon_stage2'}:
        raise ValueError('Both baseline and two-stage SupCon recipes are required')
    if config.get('metric_aggregation') not in (None, CLASS_FIRST):
        raise ValueError('Unknown metric aggregation')
    if config.get('metric_aggregation'):
        for stage in ('baseline', 'supcon_stage2'):
            config['recipes'][stage]['metric_aggregation'] = config['metric_aggregation']
    output = reduction_output(config)
    output.mkdir(parents=True, exist_ok=True)
    with run_lock(output / 'initialization'):
        manifest_path = output / 'manifest.json'
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest['hash'] != digest({k: v for k, v in manifest.items() if k != 'hash'}):
                raise ValueError('Manifest checksum mismatch')
            if set(manifest['sources']) != set(config['exercises']):
                raise ValueError('Exercise set changed')
            verify_sources(config['source_root'], manifest)
        else:
            manifest = create_manifest(config['source_root'], config['exercises'])
            write_json(manifest_path, manifest)
        folds = folds_for(config, manifest)
        model_dir = Path(config['moment_model_dir'])
        model_files = sorted(p for p in model_dir.iterdir() if p.is_file() and
                             p.suffix in ('.bin', '.safetensors', '.json', '.pt', '.pth'))
        if not model_files:
            raise ValueError(f'No local MOMENT model files: {model_dir}')
        model_hash = digest({p.name: file_digest(p) for p in model_files})
        # Machine-specific locations are not scientific hyperparameters. The
        # actual source/model contents are fingerprinted instead of their paths.
        settings = {k: v for k, v in config.items() if k not in
                    ('source_root', 'moment_model_dir', 'output_dir', 'prepared_data_dir', 'prepared_data_dirs', 'recipe_sources')}
        snapshot = dict(workflow_version=2, config=settings, manifest_hash=manifest['hash'],
                        model_hash=model_hash, folds=folds, code_hash=source_code_hash())
        protocol_hash = digest(snapshot)
        snapshot['protocol_hash'] = protocol_hash
        snapshot_path = output / 'protocol.json'
        if snapshot_path.exists() and read_json(snapshot_path) != snapshot:
            raise ValueError('Configuration, source, model, or code changed; use a new output_dir')
        write_json(snapshot_path, snapshot)
        write_json(output / 'folds.json', folds)
    return config, manifest, output, protocol_hash


def data_path(output, length, portion, kind='embeddings'):
    return output / 'data' / f'L{length}' / portion / kind


def reused_development(config, manifest, output, length):
    roots = config.get('prepared_data_dirs', [])
    if config.get('prepared_data_dir'):
        roots = [*roots, config['prepared_data_dir']]
    for root in roots:
        found = _reused_development(dict(config, prepared_data_dir=root), manifest, output, length)
        if found is not None:
            return found
    return None


def _reused_development(config, manifest, output, length):
    """Reuse completed pre-fold embeddings read-only after checking their recipe.

    Old LOPO training/checkpoints are NEVER reused. Only frozen embeddings whose
    temporal split, source/model fingerprints and preprocessing agree qualify.
    """
    if not config.get('prepared_data_dir'):
        return None
    old_root = reduction_output(dict(config, output_dir=config['prepared_data_dir']))
    path = data_path(old_root, length, 'development')
    if not path.exists() or not (old_root / 'protocol.json').exists():
        return None
    old = read_json(old_root / 'protocol.json')
    current = read_json(output / 'protocol.json')
    if old.get('manifest_hash') != manifest['hash'] or old.get('model_hash') != current['model_hash']:
        raise ValueError('Existing embedding sources/model do not match the new run')
    try:
        index, actual = store_index(path)
    except ValueError:
        print(f'Incomplete old cache; will build a new cache: {path}', flush=True)
        return None
    raw = dict(protocol_hash=old['protocol_hash'], manifest_hash=manifest['hash'],
               portion='development', length=length, stride=config['stride'],
               aggregation='any', normalization='per_frame_hip_shoulder_then_div100',
               short_samples='pad_each_contiguous_run', schema=1)
    expected = embedding_provenance(raw, raw, config.get('reduction', 'none'))
    if actual != expected or {m['pid'] for m in index} != set(manifest['patients']):
        raise ValueError(f'Existing embedding preprocessing/membership mismatch: {path}')
    return path


def development_path(config, manifest, output, length):
    local = data_path(output, length, 'development')
    if local.exists():
        _, provenance = store_index(local)
        if provenance['protocol_hash'] != read_json(output / 'protocol.json')['protocol_hash']:
            raise ValueError('Local embedding protocol mismatch')
        return local
    reused = reused_development(config, manifest, output, length)
    if reused is None:
        raise FileNotFoundError(f'Run prepare first: missing development embeddings for L{length}')
    return reused


def prepare_portions(config, manifest, output, protocol_hash, lengths, portion, device):
    if portion == 'test':
        selection = load_locked(output / 'selection.json', protocol_hash)
        final = load_locked(output / 'final_models.json', protocol_hash)
        if final['selection_hash'] != selection['hash']:
            raise ValueError('Final models and selection do not match')
        for method in ('baseline', 'supcon'):
            for seed in config['seeds']:
                run = resolve_run(final['runs'][method][str(seed)], output)
                expected = identity(protocol_hash, method, selection['methods'][method]['length'],
                                    None, seed, 'final', reduction=config.get('reduction', 'none'))
                if run['identity'] != expected or run['epochs'] != selection['methods'][method]['epochs']:
                    raise ValueError('Incomplete or incorrect final model set')
                if file_digest(run['checkpoint']) != run['checkpoint_sha256']:
                    raise ValueError('Final checkpoint changed before opening test data')
        if set(lengths) != {selection['methods'][m]['length'] for m in ('baseline', 'supcon')}:
            raise ValueError('Only selected window lengths may touch test portions')
    elif portion != 'development':
        raise ValueError(portion)
    moment = None
    for length in sorted(lengths):
        if portion == 'development' and not data_path(output, length, portion).exists():
            reused = reused_development(config, manifest, output, length)
            if reused is not None:
                print(f'Reusing completed development embeddings: {reused}', flush=True)
                continue
        provenance = dict(protocol_hash=protocol_hash, manifest_hash=manifest['hash'],
                          portion=portion, length=length, stride=config['stride'],
                          aggregation='any', normalization='per_frame_hip_shoulder_then_div100',
                          short_samples='pad_each_contiguous_run', schema=1)
        raw = data_path(output, length, portion, 'raw')
        print(f'Preparing {portion} L{length}', flush=True)
        index = write_store(raw, iter_windows(config['source_root'], manifest, portion,
                                              length, config['stride']), provenance)
        present = {m['pid'] for m in index}
        if present != set(manifest['patients']):
            raise ValueError(f'Missing patients in {portion}, L{length}: {set(manifest["patients"]) - present}')
        represented = {m['trial_id'] for m in index}
        write_json(raw.parent / 'coverage.json', dict(
            samples=len(index), patients=sorted(present), represented_trials=len(represented),
            padded_samples=sum(m['valid_length'] < length for m in index),
            trials_without_samples=[t['trial_id'] for t in manifest['trials'] if t['trial_id'] not in represented]))
        target = data_path(output, length, portion)
        if not target.exists() and moment is None:
            print(f'Loading local MOMENT, then moving to {device}...', flush=True)
            moment = load_moment(config['moment_model_dir'], device)
            print('MOMENT ready', flush=True)
        if target.exists():
            _, actual = store_index(target)
            if actual != embedding_provenance(provenance, provenance, config.get('reduction', 'none')):
                raise ValueError('Existing embedding provenance mismatch')
        else:
            embed_store(raw, target, moment, device, provenance,
                        batch_size=config.get('embedding_batch_size', 4),
                        reduction=config.get('reduction', 'none'))
        print(f'Completed {portion} L{length}: {len(index)} samples', flush=True)
    del moment
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def load_locked(path, protocol_hash):
    result = read_json(path)
    if result.get('protocol_hash') != protocol_hash:
        raise ValueError(f'Protocol mismatch: {path}')
    if result.get('hash') != digest({k: v for k, v in result.items() if k != 'hash'}):
        raise ValueError(f'Artifact checksum mismatch: {path}')
    return result


def lock_artifact(path, payload):
    payload = copy.deepcopy(payload)
    payload['hash'] = digest(payload)
    if Path(path).exists() and read_json(path) != payload:
        raise ValueError(f'Cannot replace a different completed artifact: {path}')
    write_json(path, payload)
    return payload


def identity(protocol_hash, method, length, patients, seed, phase='cv', reduction='none', fold_id=None):
    patients = [] if patients is None else ([patients] if isinstance(patients, int) else list(patients))
    return dict(protocol_hash=protocol_hash, method=method, length=length, fold_id=fold_id,
                held_out_patients=sorted(patients), seed=seed, phase=phase, reduction=reduction)


def initialization(run):
    return dict(path=run['checkpoint'], signature=run['signature'], epochs=run['epochs'])


def tasks_for(config, manifest, output, protocol_hash, method, length, stage, params,
              directory, validate=True, epochs=None, pretrained=None, phase='cv'):
    folds = folds_for(config, manifest) if phase == 'cv' else [dict(fold_id=None, validation_patients=[])]
    tasks = []
    path = str(development_path(config, manifest, output, length))
    for fold in folds:
        fid = str(fold['fold_id'])
        for seed in config['seeds']:
            ident = identity(protocol_hash, method, length, fold['validation_patients'], seed, phase,
                             config.get('reduction', 'none'), fold['fold_id'])
            tag = f'{fid}/{seed}'
            init = initialization(pretrained[tag]) if pretrained else None
            tasks.append(dict(tag=tag, params=params, stage=stage,
                              data_path=path,
                              validate=validate, identity=ident,
                              output_dir=str((directory / f'fold_{fid}' if phase == 'cv' else directory)
                                             / f'seed_{seed}'),
                              epochs=epochs, initialization=init))
    return tasks


def grouped_runs(results, folds, seeds):
    return {str(f['fold_id']): {str(s): results[f"{f['fold_id']}/{s}"] for s in seeds} for f in folds}


def select(config, manifest, output, protocol_hash, device):
    if (output / 'selection.json').exists():
        return load_locked(output / 'selection.json', protocol_hash)
    folds = folds_for(config, manifest)
    candidates = {'baseline': {}, 'supcon': {}}
    for length in sorted(config['window_lengths']):
        directory = output / 'cv' / f'L{length}'
        def execute(method, stage, params, name, **kwargs):
            return dispatch(tasks_for(config, manifest, output, protocol_hash, method, length,
                                      stage, params, directory / name, **kwargs), device)
        def aggregate(results, metric):
            return average_fold_curves(grouped_runs(results, folds, config['seeds']), folds,
                                       manifest['patients'], config['seeds'], metric, config.get('metric_aggregation'))
        baseline = execute('baseline', 'baseline', config['recipes']['baseline'], 'baseline')
        candidates['baseline'][str(length)] = aggregate(baseline, 'ba')
        stage1 = execute('supcon', 'supcon', config['recipes']['supcon_stage1'], 'supcon_curves')
        stage1_selection = aggregate(stage1, 'loss')
        pretrained = execute('supcon', 'supcon', config['recipes']['supcon_stage1'], 'supcon_selected',
                             validate=False, epochs=stage1_selection['epoch'])
        stage2 = execute('supcon', 'head', config['recipes']['supcon_stage2'], 'head_curves', pretrained=pretrained)
        selected = aggregate(stage2, 'ba')
        selected['stage1'] = stage1_selection
        candidates['supcon'][str(length)] = selected
        write_json(directory / 'validation_summary.json', {m: candidates[m][str(length)] for m in candidates})
    methods = {}
    for method, results in candidates.items():
        length = min(results, key=lambda n: (-results[n]['score'], int(n)))
        result = results[length]
        methods[method] = dict(length=int(length), epochs=result['epoch'], score=result['score'])
        if method == 'supcon':
            methods[method]['stage1_epochs'] = result['stage1']['epoch']
    return lock_artifact(output / 'selection.json', dict(
        protocol_hash=protocol_hash, reduction=config.get('reduction', 'none'), methods=methods, candidates=candidates,
        folds=folds, aggregation='seeds_within_patient_then_equal_patients_at_each_epoch',
        tie_break='earliest_epoch_then_shortest_window', test_accessed=False))


def fit(config, manifest, output, protocol_hash, device):
    selection = load_locked(output / 'selection.json', protocol_hash)
    if (output / 'final_models.json').exists():
        return load_locked(output / 'final_models.json', protocol_hash)
    runs = {}
    for method in ('baseline', 'supcon'):
        chosen = selection['methods'][method]
        length = chosen['length']
        directory = output / 'final' / method / f'L{length}'
        pretrained = None
        if method == 'supcon':
            pretrained = dispatch(tasks_for(
                config, manifest, output, protocol_hash, method, length, 'supcon',
                config['recipes']['supcon_stage1'], directory / 'stage1', validate=False,
                epochs=chosen['stage1_epochs'], phase='final'), device)
        params = config['recipes']['baseline' if method == 'baseline' else 'supcon_stage2']
        results = dispatch(tasks_for(config, manifest, output, protocol_hash, method, length,
                                     'baseline' if method == 'baseline' else 'head', params, directory / 'model',
                                     validate=False, epochs=chosen['epochs'], pretrained=pretrained, phase='final'), device)
        runs[method] = {str(seed): portable_run(results[f'None/{seed}'], output) for seed in config['seeds']}
    return lock_artifact(output / 'final_models.json', dict(
        protocol_hash=protocol_hash, reduction=config.get('reduction', 'none'),
        selection_hash=selection['hash'], runs=runs))


def test(config, manifest, output, protocol_hash, device):
    selection = load_locked(output / 'selection.json', protocol_hash)
    final = load_locked(output / 'final_models.json', protocol_hash)
    if final['selection_hash'] != selection['hash']:
        raise ValueError('Final models were fit using a different selection')
    if (output / 'test_results.json').exists():
        return load_locked(output / 'test_results.json', protocol_hash)
    lengths = {selection['methods'][m]['length'] for m in ('baseline', 'supcon')}
    prepare_portions(config, manifest, output, protocol_hash, lengths, 'test', device)
    results = {}
    for method in ('baseline', 'supcon'):
        length = selection['methods'][method]['length']
        store = EmbeddingStore(data_path(output, length, 'test'))
        try:
            seed_results = {}
            for seed in config['seeds']:
                run = resolve_run(final['runs'][method][str(seed)], output)
                path = output / 'test' / method / f'seed_{seed}.json'
                if path.exists():
                    result = read_json(path)
                    if result['signature'] != run['signature']:
                        raise ValueError('Test prediction checkpoint mismatch')
                else:
                    result = predict_final(config['recipes']['baseline' if method == 'baseline' else 'supcon_stage2'],
                                           'baseline' if method == 'baseline' else 'head', store, run,
                                           manifest['patients'], path, device)
                seed_results[str(seed)] = result
            results[method] = aggregate_test(seed_results, manifest['patients'], config['seeds'], config.get('metric_aggregation'))
            results[method]['pooled_ba_per_seed'] = {s: r['pooled']['ba'] for s, r in seed_results.items()}
        finally:
            store.close()
    return lock_artifact(output / 'test_results.json', dict(
        protocol_hash=protocol_hash, reduction=config.get('reduction', 'none'), selection_hash=selection['hash'],
        final_models_hash=final['hash'], methods=results))

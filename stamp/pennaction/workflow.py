"""Isolated PennAction orchestration; does not alter SERE protocols or outputs."""
import csv
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from stamp.sere.data import digest, file_digest, read_json, write_json
from stamp.sere.checkpoint import run_lock
from .data import build_manifest
from .embedding import prepare
from .training import fit_task, test_model

_DEVICE = None


def worker_init(queue):
    global _DEVICE
    _DEVICE = queue.get()
    torch.set_num_threads(2)
    if torch.device(_DEVICE).type == 'cuda':
        torch.cuda.set_device(_DEVICE)


def worker(task):
    return fit_task(task, _DEVICE)


def dispatch(tasks, devices):
    if len(set(devices)) != len(devices):
        raise ValueError('Each worker needs a distinct device')
    for device in devices:
        parsed = torch.device(device)
        if parsed.type == 'cuda' and (not torch.cuda.is_available() or (parsed.index or 0) >= torch.cuda.device_count()):
            raise ValueError(f'GPU is not visible: {device}')
    if len(devices) == 1:
        return [fit_task(task, devices[0]) for task in tasks]
    context = mp.get_context('spawn')
    queue = context.Queue()
    for device in devices:
        queue.put(device)
    try:
        with context.Pool(len(devices), initializer=worker_init, initargs=(queue,)) as pool:
            return list(pool.imap_unordered(worker, tasks))
    finally:
        queue.close()


def validate_config(config):
    if config['seeds'] != [654, 114, 25, 759, 281]:
        raise ValueError('PennAction uses the existing five STAMP seeds')
    if config['patch_len'] < 1 or config['padded_frames'] < 1 or config['padded_frames'] % config['patch_len']:
        raise ValueError('Padding capacity must be patch aligned')
    if set(config['recipes']) != {'baseline', 'supcon_stage1', 'supcon_stage2'}:
        raise ValueError('Baseline and both SupCon stages are required')


def initialize(config, reduction):
    validate_config(config)
    if reduction not in ('none', 'mean'):
        raise ValueError(reduction)
    root = Path(config['output_dir']).resolve() / ('without_reduction' if reduction == 'none' else 'with_reduction')
    manifest = build_manifest(config)
    model_dir = Path(config['moment_model_dir'])
    models = sorted(p for p in model_dir.iterdir() if p.suffix in ('.json', '.bin', '.safetensors', '.pt', '.pth'))
    if not models:
        raise ValueError('No local MOMENT files')
    package = Path(__file__).parent
    # Fingerprint imported scientific helpers as well as this isolated package.
    files = sorted(package.glob('*.py')) + sorted((package.parent / 'modeling').glob('*.py'))
    files += [package.parent / 'sere' / f'{name}.py' for name in ('data', 'embedding', 'training', 'checkpoint')]
    snapshot = dict(protocol='pennaction-whole-video-v1', reduction=reduction,
        config={k: v for k, v in config.items() if k not in ('output_dir', 'source_root', 'split_lmdb', 'moment_model_dir')},
        manifest_hash=manifest['hash'], model_hash=digest({p.name: file_digest(p) for p in models}),
        code_hash=digest({str(p.relative_to(package.parent)): file_digest(p) for p in files}))
    signature = digest(snapshot)
    snapshot['protocol_hash'] = signature
    with run_lock(root / 'initialization'):
        path = root / 'protocol.json'
        if path.exists() and read_json(path) != snapshot:
            raise ValueError('Configuration/data/code changed; choose a NEW PennAction output_dir')
        write_json(path, snapshot)
        write_json(root / 'manifest.json', manifest)
    return root, manifest, signature


def train(config, manifest, root, signature, reduction, devices):
    results = {}
    pretrained = {}
    for stage in ('baseline', 'supcon', 'head'):
        tasks = [dict(config=config, manifest=manifest, root=str(root), protocol_hash=signature,
                      reduction=reduction, stage=stage, seed=seed,
                      run_dir=str(root / 'runs' / stage / f'seed_{seed}'),
                      initialization=pretrained.get(seed) if stage == 'head' else None)
                 for seed in config['seeds']]
        runs = dispatch(tasks, devices)
        if stage == 'supcon':
            pretrained = {r['seed']: r for r in runs}
        else:
            results['baseline' if stage == 'baseline' else 'supcon'] = {str(r['seed']): r for r in runs}
    result = dict(protocol_hash=signature, methods=results)
    write_json(root / 'trained_models.json', result)
    return result


def summarize(seed_results):
    output = {}
    for key in ('validation_ba', 'validation_accuracy', 'test_ba', 'test_accuracy'):
        values = [r[key] for r in seed_results]
        output[f'{key}_mean'] = float(np.mean(values))
        output[f'{key}_std'] = float(np.std(values, ddof=1))
    return output


def test(config, manifest, root, signature, reduction, device):
    final = read_json(root / 'trained_models.json')
    if final['protocol_hash'] != signature or set(final['methods']) != {'baseline', 'supcon'}:
        raise ValueError('Finish training both methods before test')
    for runs in final['methods'].values():
        if set(runs) != {str(s) for s in config['seeds']}:
            raise ValueError('Finish ALL five seeds before test')
        for run in runs.values():
            if file_digest(root / run['checkpoint']) != run['checkpoint_sha256']:
                raise ValueError('Selected checkpoint missing or changed')
    prepare(config, manifest, root, signature, reduction, ['test'], device)
    seed_rows, rows = [], []
    for method, runs in final['methods'].items():
        current = []
        for seed in config['seeds']:
            run = runs[str(seed)]
            path = root / 'test' / method / f'seed_{seed}.json'
            if path.exists():
                result = read_json(path)
                if result['signature'] != run['signature'] or result['checkpoint_sha256'] != run['checkpoint_sha256']:
                    raise ValueError('Test results refer to another checkpoint')
            else:
                result = dict(signature=run['signature'], checkpoint_sha256=run['checkpoint_sha256'],
                              metrics=test_model(config, manifest, root, signature, reduction, run, device))
                write_json(path, result)
            measured = result['metrics']
            current.append(dict(reduction=reduction, method=method, seed=seed, best_epoch=run['best_epoch'],
                validation_ba=run['validation']['ba'], validation_accuracy=run['validation']['accuracy'],
                test_ba=measured['ba'], test_accuracy=measured['accuracy']))
        seed_rows.extend(current)
        rows.append(dict(reduction=reduction, method=method, seeds=len(current), std_ddof=1, **summarize(current)))
    result = dict(protocol_hash=signature, unit='video', rows=rows, per_seed=seed_rows)
    write_json(root / 'evaluation_results.json', result)
    for name, data in [('evaluation_results', rows), ('per_seed_results', seed_rows)]:
        path = root / f'{name}.csv'
        temp = path.with_suffix('.csv.tmp')
        with temp.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        temp.replace(path)
    return result

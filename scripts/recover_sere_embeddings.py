"""Rebuild deleted legacy development caches without changing saved run identities.

Training/model code remains unchanged. Historical cache provenance is recovered
from saved requests; source/model/code identity and sample membership are checked.
This regenerates embeddings, not a byte-for-byte backup restore. Different GPU
hardware or library versions can produce small floating-point differences.
"""
import argparse
from pathlib import Path
import shutil

from stamp.sere.checkpoint import run_lock
from stamp.sere.data import (digest, fold_keys, iter_windows, read_json, store_index,
                            write_json, write_store)
from stamp.sere.embedding import embed_store, embedding_provenance, load_moment
from stamp.sere.workflow import data_path, initialize, reduction_output


def recovery_plan(config, manifest, output, protocol_hash):
    requests = [(p, read_json(p)) for p in sorted(output.glob('cv/**/request.json'))]
    # Newly added windows prepared in the current output have current-protocol
    # caches and do not belong to this historical-cache recovery operation.
    requests = [(p, r) for p, r in requests if r['data']['protocol_hash'] != protocol_hash]
    if not requests:
        raise ValueError('No saved CV requests to recover; use ordinary prepare for a new experiment')
    old_hashes = {request['data']['protocol_hash'] for _, request in requests}
    if len(old_hashes) != 1:
        raise ValueError('Expected one historical cache identity across saved requests')
    old_hash = old_hashes.pop()
    if old_hash == protocol_hash:
        raise ValueError('These runs used local five-fold caches, not the deleted legacy caches')
    raw_by_length = {}
    for length in config['window_lengths']:
        raw_by_length[length] = dict(
            protocol_hash=old_hash, manifest_hash=manifest['hash'], portion='development',
            length=length, stride=config['stride'], aggregation='any',
            normalization='per_frame_hip_shoulder_then_div100',
            short_samples='pad_each_contiguous_run', schema=1)
    for path, request in requests:
        ident = request['identity']
        if ident['protocol_hash'] != protocol_hash or ident['reduction'] != config['reduction']:
            raise ValueError(f'Saved request belongs to another protocol: {path}')
        raw = raw_by_length[ident['length']]
        if request['data'] != embedding_provenance(raw, raw, config['reduction']):
            raise ValueError(f'Historical preprocessing differs from current recipe: {path}')
        result_path = path.with_name('run.json')
        if result_path.exists() and read_json(result_path)['signature'] != digest(request):
            raise ValueError(f'Saved result/request signature mismatch: {path}')
    return requests, raw_by_length


def verify_membership(index, requests, length, patients):
    if {row['pid'] for row in index} != set(patients):
        raise ValueError(f'Rebuilt L{length} patient membership differs')
    for path, request in requests:
        if request['identity']['length'] != length:
            continue
        held = request['identity']['held_out_patients']
        for role in ('train', 'val'):
            # Stage-1 replay to the selected epoch has no validation loader.
            if role == 'val' and request['val_keys_hash'] is None:
                continue
            if digest(fold_keys(index, held, role)) != request[f'{role}_keys_hash']:
                raise ValueError(f'Rebuilt {role} sample keys differ from saved run: {path}')


def complete_store(target, provenance, build):
    """Publish only complete stores; retry only our own interrupted staging tree."""
    target = Path(target)
    if target.exists():
        index, actual = store_index(target)
        if actual != provenance:
            raise ValueError(f'Existing cache provenance differs: {target}')
        return index
    staging = target.with_name(target.name + '.rebuilding')
    if staging.exists():
        try:
            index, actual = store_index(staging)
        except ValueError:
            print(f'Restarting incomplete recovery cache: {staging}', flush=True)
            shutil.rmtree(staging)
        else:
            if actual != provenance:
                raise ValueError(f'Staging cache belongs to another recovery: {staging}')
            staging.rename(target)
            return index
    build(staging)
    index, actual = store_index(staging)
    if actual != provenance:
        raise ValueError(f'Rebuilt cache provenance differs: {staging}')
    staging.rename(target)
    return index


def recover(config_path, reduction, device, raw_only=False):
    # Works both as a command and when imported by tests.
    if __package__:
        from .sere_selection_policy import apply_selection_policy
    else:
        from sere_selection_policy import apply_selection_policy
    # initialize verifies the original scientific config, code, sources and
    # MOMENT checksums, while permitting changed physical storage paths.
    config, manifest, output, protocol_hash = initialize(config_path, reduction)
    if not config.get('prepared_data_dir'):
        raise ValueError('prepared_data_dir must point to the recovery location')
    target_root = reduction_output(dict(config, output_dir=config['prepared_data_dir']))
    if target_root == output:
        raise ValueError('Recovery cache and training output directories must differ')
    with run_lock(output / 'orchestrator'), run_lock(target_root / 'recovery'):
        requests, raw_by_length = recovery_plan(config, manifest, output, protocol_hash)
        # Validate all historical request recipes before restricting regeneration.
        config = apply_selection_policy(config, config_path, output, protocol_hash)
        new_lengths = set(config['window_lengths']) - set(raw_by_length)
        if new_lengths:
            print(f'New lengths {sorted(new_lengths)} use ordinary prepare, not legacy recovery.', flush=True)
        raw_by_length = {n: raw_by_length[n] for n in config['window_lengths'] if n in raw_by_length}
        if not raw_by_length:
            print('No historical lengths to recover; run prepare for the new lengths.', flush=True)
            return
        current = read_json(output / 'protocol.json')
        # This is explicitly a reconstructed cache descriptor, not the original
        # lost protocol snapshot. The cache identity comes from saved requests.
        descriptor = dict(protocol_hash=next(iter(raw_by_length.values()))['protocol_hash'],
                          manifest_hash=manifest['hash'], model_hash=current['model_hash'])
        old_protocol = target_root / 'protocol.json'
        if old_protocol.exists():
            existing = read_json(old_protocol)
            if any(existing.get(k) != v for k, v in descriptor.items()):
                raise ValueError('Existing cache descriptor differs; refusing to overwrite')
        else:
            write_json(old_protocol, dict(descriptor, reconstructed_from_saved_requests=True))
        write_json(target_root / 'manifest.json', manifest)
        audit = dict(protocol_hash=protocol_hash, historical_cache=descriptor,
                     request_hashes={str(p.relative_to(output)): digest(r) for p, r in requests},
                     note='Regenerated from verified sources/model; not a bitwise backup restore.',
                     device=device, embedding_batch_size=config.get('embedding_batch_size', 4))
        write_json(target_root / 'recovery.json', audit)
        moment = None
        for length, raw_provenance in raw_by_length.items():
            raw = data_path(target_root, length, 'development', 'raw')
            print(f'Recovering development L{length}: {target_root}', flush=True)
            def build_raw(path):
                records = iter_windows(config['source_root'], manifest, 'development',
                                       length, config['stride'])
                write_store(path, records, raw_provenance)
            index = complete_store(raw, raw_provenance, build_raw)
            verify_membership(index, requests, length, manifest['patients'])
            checked = sum(r['identity']['length'] == length for _, r in requests)
            print(f'Validated L{length}: {len(index)} samples; {checked} saved requests checked', flush=True)
            if raw_only:
                continue
            target = data_path(target_root, length, 'development')
            expected = embedding_provenance(raw_provenance, raw_provenance, reduction)
            def build_embeddings(path):
                nonlocal moment
                if moment is None:
                    print(f'Loading MOMENT on {device}', flush=True)
                    moment = load_moment(config['moment_model_dir'], device)
                embed_store(raw, path, moment, device, raw_provenance,
                            batch_size=config.get('embedding_batch_size', 4), reduction=reduction)
            embedded_index = complete_store(target, expected, build_embeddings)
            if embedded_index != index:
                raise ValueError(f'Embedding metadata differs from raw samples: L{length}')
            print(f'Recovered L{length} embeddings', flush=True)
        print('Raw data verified; run again without --raw-only to rebuild embeddings.' if raw_only
              else 'Development embeddings recovered. Resume selection with the usual command.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/sere_temporal_lopo.json')
    parser.add_argument('--reduction', choices=['none', 'mean'], default='none')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--raw-only', action='store_true', help='Rebuild and validate raw samples without a GPU')
    args = parser.parse_args()
    recover(args.config, args.reduction, args.device, args.raw_only)


if __name__ == '__main__':
    main()

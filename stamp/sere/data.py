"""Original-frame temporal partitions and indexed, masked LMDB samples.

No random patient split is used here. The manifest is independent of window
length and labels; validation folds are views of development data only.
"""
import hashlib
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd

PROTOCOL = "sere-temporal80-lopo-v1"
SEEDS = [654, 114, 25, 759, 281]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_paths(root, exercise):
    root = Path(root)
    return {
        'world': root / 'MediaPipe_skeletons' / 'WorldLandmarks' / f'{exercise}_mp_world_landmarks.csv',
        'frames': root / 'Labels' / 'frame_level' / 'compensation' / f'{exercise}_frame_labels.csv',
        'trials': root / 'Labels' / 'video_level' / 'compensation' / f'{exercise}_labels_comp.csv',
    }


def read_table(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df


def create_manifest(root, exercises):
    trials, sources = [], {}
    for ex in exercises:
        paths = source_paths(root, ex)
        sources[ex] = {name: file_digest(path) for name, path in paths.items()}
        df = read_table(paths['trials'])
        # SERE E3/E4/E5 use the sentinel -1 in frame and skeleton tables.
        if 'affected' not in df:
            df['affected'] = -1
        df['affected'] = df['affected'].fillna(-1)
        for row in df.to_dict('records'):
            start, end = int(row['frame_init']), int(row['frame_end'])
            if end < start:
                raise ValueError(f'Invalid trial interval: {ex} {row}')
            pid, affected, trial = int(row['pid']), int(row['affected']), int(row['trial'])
            trials.append(dict(
                trial_id=f'{ex}_pid{pid}_aff{affected}_trial{trial}',
                exercise=ex, pid=pid, affected=affected, trial=trial,
                start=start, stop=end + 1, cut=start + (4 * (end - start + 1)) // 5,
                excluded_frames=[],
            ))
    trials.sort(key=lambda t: (t['exercise'], t['pid'], t['affected'], t['trial']))
    if len({t['trial_id'] for t in trials}) != len(trials):
        raise ValueError('Duplicate trial IDs')
    groups = defaultdict(list)
    for t in trials:
        groups[(t['exercise'], t['pid'], t['affected'])].append(t)
    for group in groups.values():
        owners = defaultdict(list)
        for t in group:
            for frame in range(t['start'], t['stop']):
                owners[frame].append(t)
        for frame, assignments in owners.items():
            if len(assignments) > 1:
                for t in assignments:
                    t['excluded_frames'].append(frame)
    manifest = dict(protocol=PROTOCOL, sources=sources, trials=trials,
                    patients=sorted({t['pid'] for t in trials}),
                    boundary_policy='floor_80pct_original_timeline_exclude_shared_frames')
    manifest['hash'] = digest(manifest)
    return manifest


def verify_sources(root, manifest):
    for ex, expected in manifest['sources'].items():
        for name, path in source_paths(root, ex).items():
            if file_digest(path) != expected[name]:
                raise ValueError(f'Source changed since manifest creation: {path}')


def contiguous_runs(frames):
    frames = np.asarray(frames, dtype=np.int64)
    if not len(frames):
        return []
    return np.split(frames, np.flatnonzero(np.diff(frames) != 1) + 1)


def window_slices(size, length, stride):
    if size <= 0:
        return []
    if size < length:
        return [(0, size)]
    return [(start, start + length) for start in range(0, size - length + 1, stride)]


def normalize_xyz(x):
    root = (x[:, 23] + x[:, 24]) * 0.5
    x = x - root[:, None, :]
    scale = np.linalg.norm(x[:, 11] - x[:, 12], axis=-1)
    return x / np.maximum(scale[:, None, None], 1e-6)


def iter_windows(root, manifest, portion, length, stride):
    if portion not in ('development', 'test') or length < 1 or stride < 1:
        raise ValueError('Invalid portion/window/stride')
    columns = [f'{j}{axis}' for j in range(33) for axis in 'xyz']
    for ex in manifest['sources']:
        paths = source_paths(root, ex)
        world, labels = read_table(paths['world']), read_table(paths['frames'])
        # E2 frame labels contain an exactly duplicated block. Deduplicate only
        # identical rows; conflicting values for one frame remain a hard error.
        world, labels = world.drop_duplicates(), labels.drop_duplicates()
        for table in (world, labels):
            if 'affected' not in table:
                table['affected'] = -1
            table['affected'] = table['affected'].fillna(-1).astype(int)
            table['pid'] = table['pid'].astype(int)
            table['frame'] = table['frame'].astype(int)
            if table.duplicated(['pid', 'affected', 'frame']).any():
                raise ValueError(f'Duplicate source frame rows in {ex}')
            table.set_index(['pid', 'affected', 'frame'], inplace=True)
        for trial in (t for t in manifest['trials'] if t['exercise'] == ex):
            lo, hi = ((trial['start'], trial['cut']) if portion == 'development'
                      else (trial['cut'], trial['stop']))
            frames = np.arange(lo, hi)
            if not len(frames):
                continue
            index = pd.MultiIndex.from_arrays([
                np.full(len(frames), trial['pid']), np.full(len(frames), trial['affected']), frames])
            x = world.reindex(index)[columns].to_numpy(dtype=np.float32).reshape(-1, 33, 3)
            y = pd.to_numeric(labels.reindex(index)['comp'], errors='coerce').to_numpy(dtype=np.float32)
            # Missing skeletons and shared boundary frames break continuity. Do
            # not concatenate the remaining frames or interpolate across gaps.
            valid = np.isfinite(x).all(axis=(1, 2)) & ~np.isin(frames, trial['excluded_frames'])
            for run in contiguous_runs(frames[valid]):
                for start, stop in window_slices(len(run), length, stride):
                    selected = run[start:stop]
                    offsets = selected - lo
                    target = y[offsets]
                    known = target[np.isfinite(target)]
                    if not len(known):
                        continue
                    values = normalize_xyz(x[offsets]).transpose(1, 2, 0).reshape(99, -1)
                    sample = np.zeros((99, length), dtype=np.float32)
                    sample[:, :len(selected)] = values / 100.0  # preserve legacy raw-loader scaling
                    mask = np.arange(length) < len(selected)
                    key = f"{trial['trial_id']}_{portion}_f{selected[0]}_n{len(selected)}_L{length}"
                    meta = dict(key=key, trial_id=trial['trial_id'], pid=trial['pid'],
                                exercise=ex, affected=trial['affected'], portion=portion,
                                frame_start=int(selected[0]), frame_stop=int(selected[-1]) + 1,
                                valid_length=len(selected), label=int(np.any(known >= 0.5)))
                    assert (meta['frame_stop'] <= trial['cut'] if portion == 'development'
                            else meta['frame_start'] >= trial['cut'])
                    yield dict(sample=sample, mask=mask, meta=meta)


def write_store(path, records, provenance, map_size=64 * 1024**3):
    """Atomic metadata completion; refuse to reuse a partial or unrelated store."""
    path = Path(path)
    if path.exists():
        index, stored = store_index(path)
        if stored != provenance:
            raise ValueError(f'Existing store has different provenance: {path}')
        return index
    path.mkdir(parents=True)
    env = lmdb.open(str(path), map_size=map_size)
    index, seen = [], set()
    try:
        txn = env.begin(write=True)
        for record in records:
            meta = record['meta']
            key = meta['key']
            if key in seen:
                raise ValueError(f'Duplicate sample key: {key}')
            seen.add(key)
            index.append(meta)
            txn.put(key.encode(), pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL))
            if len(index) % 100 == 0:
                txn.commit()
                txn = env.begin(write=True)
        if not index:
            raise ValueError(f'No usable samples for {path}')
        txn.put(b'__index__', json.dumps(index).encode())
        txn.put(b'__provenance__', json.dumps(provenance, sort_keys=True).encode())
        txn.commit()
    finally:
        env.close()
    return index


def store_index(path):
    env = lmdb.open(str(path), readonly=True, lock=False, create=False)
    try:
        with env.begin() as txn:
            index, provenance = txn.get(b'__index__'), txn.get(b'__provenance__')
            if index is None or provenance is None:
                raise ValueError(f'Incomplete LMDB (use a new output directory): {path}')
            return json.loads(index), json.loads(provenance)
    finally:
        env.close()


def fold_keys(index, held_out_patient=None, role='train'):
    if role not in ('train', 'val', 'final', 'test'):
        raise ValueError(role)
    expected = 'test' if role == 'test' else 'development'
    if any(m['portion'] != expected for m in index):
        raise ValueError(f'{role} may only load {expected} data')
    if role in ('train', 'val') and held_out_patient is None:
        raise ValueError('Grouped CV requires held-out patients')
    held_out = set(held_out_patient if isinstance(held_out_patient, (list, tuple, set))
                   else [held_out_patient])
    selected = [m['key'] for m in index if role in ('final', 'test') or
                ((m['pid'] in held_out) == (role == 'val'))]
    if not selected:
        raise ValueError(f'Empty {role} set for patient {held_out_patient}')
    return selected


def patient_folds(patients, n_folds=5, seed=42):
    """Deterministic patient groups, independent of windows, labels, and run seeds."""
    patients = sorted(patients)
    if len(set(patients)) != len(patients) or not 2 <= n_folds <= len(patients):
        raise ValueError('Need distinct patients and 2 <= n_folds <= patient count')
    shuffled = patients.copy()
    random.Random(seed).shuffle(shuffled)
    folds = []
    for i in range(n_folds):
        validation = sorted(shuffled[i::n_folds])
        folds.append(dict(fold_id=i + 1, validation_patients=validation,
                          training_patients=[p for p in patients if p not in validation]))
    return folds

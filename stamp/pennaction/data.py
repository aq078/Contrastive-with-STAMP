"""PennAction split import and full-video skeleton normalization (no windows)."""
from pathlib import Path
import pickle

import lmdb
import numpy as np
from scipy.io import loadmat

from stamp.sere.data import digest, file_digest


def supplied_split(path):
    env = lmdb.open(str(path), readonly=True, lock=False, create=False)
    try:
        with env.begin() as txn:
            def get(name):
                value = txn.get(name.encode())
                if value is None:
                    raise ValueError(f'Missing {name} in split LMDB: {path}')
                return pickle.loads(value)
            split, classes, metadata, keys = (get(k) for k in
                ('__sequence_split__', '__class_to_idx__', '__dataset_meta__', '__keys__'))
            if set(keys) != {'train', 'val', 'test'} or set(split.values()) != set(keys):
                raise ValueError('Expected provided train/val/test video assignments')
            observed = {}
            for portion, entries in keys.items():
                for key in entries:
                    video = key.split('_w')[0]
                    if video in observed and observed[video] != portion:
                        raise ValueError(f'Video occurs in multiple splits: {video}')
                    observed[video] = portion
            if observed != split:
                raise ValueError('Window membership disagrees with provided video split')
            if sorted(classes.values()) != list(range(len(classes))):
                raise ValueError('Class IDs must be contiguous from zero')
            return split, classes, metadata.get('all_skipped_sequence_ids', [])
    finally:
        env.close()


def annotation(path):
    return loadmat(path, simplify_cells=True)


def build_manifest(config):
    split, classes, excluded = supplied_split(config['split_lmdb'])
    paths = {p.stem: p for p in (Path(config['source_root']) / 'labels').glob('*.mat')}
    if set(paths) != set(split) | set(excluded) or set(split) & set(excluded):
        raise ValueError('Source videos differ from supplied split and recorded exclusions')
    videos = []
    for video, portion in sorted(split.items()):
        raw = annotation(paths[video])
        frames = int(raw['nframes'])
        flag = int(raw['train'])
        if (portion == 'test') != (flag != 1):
            raise ValueError(f'Official train/test flag disagrees for {video}')
        if not 1 <= frames <= config['padded_frames']:
            raise ValueError(f'{video} has {frames} frames; increase padding capacity, never truncate')
        videos.append(dict(video=video, split=portion, label=int(classes[str(raw['action'])]),
                           frames=frames, sha256=file_digest(paths[video])))
    for portion in ('train', 'val', 'test'):
        if {r['label'] for r in videos if r['split'] == portion} != set(classes.values()):
            raise ValueError(f'{portion} does not contain every class')
    result = dict(classes=classes, excluded=sorted(excluded), videos=videos,
                  split_hash=digest(split), unit='whole_video')
    result['hash'] = digest(result)
    return result


def normalized_video(path):
    raw = annotation(path)
    xy = np.stack([raw['x'], raw['y']], axis=-1).astype(np.float32)
    if xy.shape != (int(raw['nframes']), 13, 2):
        raise ValueError(f'Malformed joint array: {path}: {xy.shape}')
    length = len(xy)
    visible = np.asarray(raw['visibility'], dtype=bool)
    if visible.shape != (length, 13):
        raise ValueError(f'Malformed visibility: {path}')
    boxes = np.asarray(raw['bbox'], dtype=np.float32).reshape(-1, 4)
    bbox = np.full((length, 4), np.nan, dtype=np.float32)
    bbox[:min(length, len(boxes))] = boxes[:length]
    valid = np.isfinite(bbox).all(1) & (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])
    for t in np.flatnonzero(~valid):
        good = visible[t] & np.isfinite(xy[t]).all(1)
        if not good.any():
            good = np.isfinite(xy[t]).all(1)
        if good.any():
            lo, hi = xy[t, good].min(0), xy[t, good].max(0)
            center = (lo + hi) / 2
            span = np.maximum(hi - lo, 1.0)
            bbox[t] = np.r_[center - span / 2, center + span / 2]
    valid = np.isfinite(bbox).all(1) & (bbox[:, 2] > bbox[:, 0]) & (bbox[:, 3] > bbox[:, 1])
    if not valid.any():
        raise ValueError(f'No usable bounding boxes: {path}')
    time = np.arange(length)
    for c in range(4):
        bbox[:, c] = np.interp(time, time[valid], bbox[valid, c])
    center = (bbox[:, :2] + bbox[:, 2:]) / 2
    for joint in range(13):
        for c in range(2):
            good = visible[:, joint] & np.isfinite(xy[:, joint, c])
            xy[:, joint, c] = (np.interp(time, time[good], xy[good, joint, c])
                               if good.any() else center[:, c])
    scale = (bbox[:, 2:] - bbox[:, :2]).max(1)
    xy = (xy - center[:, None, :]) / scale[:, None, None]
    if not np.isfinite(xy).all():
        raise ValueError(f'Nonfinite normalized coordinates: {path}')
    # Joint-major scalar series: j0x,j0y,j1x,j1y,... . No extra division by 100.
    return xy.transpose(1, 2, 0).reshape(26, length).copy()

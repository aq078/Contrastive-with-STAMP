"""Frozen MOMENT per-video caches, resumable at video boundaries."""
from pathlib import Path

import torch
from torch.utils.data import Dataset

from stamp.sere.checkpoint import atomic_torch_save
from stamp.sere.data import file_digest, read_json, write_json
from stamp.sere.embedding import load_moment, masked_moment_embeddings
from .data import normalized_video


def reduce_video(features, valid, reduction):
    if features.ndim != 3 or features.shape[1] != 26 or valid.shape != features.shape[:2]:
        raise ValueError('Expected video patches (P,26,D) and mask (P,26)')
    if reduction == 'none':
        return features, valid
    if reduction != 'mean':
        raise ValueError(reduction)
    counts = valid.sum(0)
    if not (counts > 0).all():
        raise ValueError('Each coordinate needs an observed patch')
    pooled = features.masked_fill(~valid[..., None], 0).sum(0) / counts[:, None]
    return (pooled.reshape(13, 2, -1).permute(1, 0, 2).contiguous(),
            (counts > 0).reshape(13, 2).T.contiguous())


def prepare(config, manifest, root, signature, reduction, portions, device):
    cache = Path(root) / 'embeddings'
    cache.mkdir(parents=True, exist_ok=True)
    model = None
    for row in manifest['videos']:
        if row['split'] not in portions:
            continue
        path = cache / f"{row['video']}.pt"
        proof = path.with_suffix('.json')
        identity = dict(protocol_hash=signature, reduction=reduction, video=row['video'],
                        source_sha256=row['sha256'], split=row['split'])
        if path.exists():
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['identity'] != identity or saved['label'] != row['label']:
                raise ValueError(f'Embedding identity mismatch: {path}')
            checksum = file_digest(path)
            if proof.exists() and read_json(proof) != dict(identity=identity, sha256=checksum):
                raise ValueError(f'Embedding checksum mismatch: {path}')
            write_json(proof, dict(identity=identity, sha256=checksum))
            continue
        source = Path(config['source_root']) / 'labels' / f"{row['video']}.mat"
        if file_digest(source) != row['sha256']:
            raise ValueError(f'Source changed: {source}')
        samples = torch.from_numpy(normalized_video(source)).unsqueeze(0)
        if model is None:
            print(f'Loading local MOMENT on {device}', flush=True)
            model = load_moment(config['moment_model_dir'], device)
            if model.patch_len != config['patch_len']:
                raise ValueError('MOMENT patch size differs from config')
        frames = samples.shape[-1]
        pad = (-frames) % model.patch_len
        samples = torch.nn.functional.pad(samples, (0, pad))
        mask = torch.arange(frames + pad)[None, :] < frames
        features, valid = masked_moment_embeddings(model, samples.to(device), mask.to(device))
        features, valid = reduce_video(features[0].cpu(), valid[0].cpu(), reduction)
        atomic_torch_save(dict(identity=identity, features=features, mask=valid,
                               label=row['label']), path)
        write_json(proof, dict(identity=identity, sha256=file_digest(path)))
        print(f"Prepared {reduction}/{row['split']}/{row['video']}: {frames} frames, one instance", flush=True)
    del model
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


class VideoDataset(Dataset):
    def __init__(self, root, manifest, split, config, signature, reduction):
        self.rows = [r for r in manifest['videos'] if r['split'] == split]
        self.cache = Path(root) / 'embeddings'
        self.signature, self.reduction = signature, reduction
        self.patches = config['padded_frames'] // config['patch_len']
        for row in self.rows:
            path = self.cache / f"{row['video']}.pt"
            proof = read_json(path.with_suffix('.json'))
            if proof['sha256'] != file_digest(path):
                raise ValueError(f'Embedding checksum mismatch: {path}')
        first = self[0][0]
        self.shape = tuple(first.shape)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        saved = torch.load(self.cache / f"{row['video']}.pt", map_location='cpu', weights_only=True)
        expected = dict(protocol_hash=self.signature, reduction=self.reduction, video=row['video'],
                        source_sha256=row['sha256'], split=row['split'])
        if saved['identity'] != expected or saved['label'] != row['label']:
            raise ValueError('Video/split/cache identity mismatch')
        x, mask = saved['features'], saved['mask']
        if self.reduction == 'none':
            if x.shape[0] > self.patches:
                raise ValueError('Video exceeds padding capacity; refusing truncation')
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, self.patches - x.shape[0]))
            mask = torch.nn.functional.pad(mask, (0, 0, 0, self.patches - mask.shape[0]))
        return x, row['label'], mask, row['video']

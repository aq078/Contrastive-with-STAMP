"""Frozen MOMENT patch embeddings, including partially observed patches."""
from pathlib import Path

import lmdb
import numpy as np
import pickle
import torch
from torch.utils.data import Dataset

from .data import digest, fold_keys, store_index, write_store


def embedding_provenance(provenance, raw_provenance, reduction):
    if reduction not in ('none', 'mean'):
        raise ValueError(f'Unknown MOMENT reduction: {reduction}')
    return dict(provenance, raw_hash=digest(raw_provenance), reduction=reduction,
                layout='3x33' if reduction == 'mean' else 'Px99',
                partial_patch_policy='any_observed_zero_normalized_padding')


def arrange_embeddings(features, valid, reduction):
    """Map MOMENT patches to STAMP axes without changing joint/coordinate order.

    Raw channels are joint-major: j0x,j0y,j0z,j1x,... . Mean reduction averages
    observed patches per scalar series, then maps [joint,coordinate] to
    STAMP's [temporal=coordinate,spatial=joint] axes. Partially observed patches
    each count once; fully padded patches do not contribute to the mean.
    """
    if features.ndim != 4 or features.shape[2] != 99 or valid.shape != features.shape[:3]:
        raise ValueError('Expected embeddings (B,P,99,D) and validity mask (B,P,99)')
    if reduction == 'none':
        return features, valid
    if reduction != 'mean':
        raise ValueError(f'Unknown MOMENT reduction: {reduction}')
    count = valid.sum(dim=1)
    if not (count > 0).all():
        raise ValueError('Every joint-coordinate series needs an observed patch')
    pooled = features.masked_fill(~valid.unsqueeze(-1), 0).sum(dim=1) / count.unsqueeze(-1)
    batch, _, dim = pooled.shape
    pooled = pooled.reshape(batch, 33, 3, dim).permute(0, 2, 1, 3).contiguous()
    mask = (count > 0).reshape(batch, 33, 3).permute(0, 2, 1).contiguous()
    return pooled, mask


@torch.no_grad()
def masked_moment_embeddings(model, samples, masks):
    """Like MOMENT.embed(reduction='none'), retaining partial final patches.

    RevIN sees only observed frames. Unobserved normalized values are zero.
    A patch is an attention key when it contains ANY observed frame, instead
    of MOMENT's default ALL rule (which loses samples shorter than 8 frames).
    """
    b, channels, length = samples.shape
    patch_len = model.patch_len
    if model.tokenizer.stride != patch_len or length % patch_len:
        raise ValueError('Expected nonoverlapping patches and patch-aligned input')
    if masks.shape != (b, length) or not masks.any(dim=1).all():
        raise ValueError('Every sample must have observed frames')
    flat = samples.reshape(b * channels, 1, length)
    frame_mask = masks.repeat_interleave(channels, dim=0)
    flat = model.normalizer(x=flat, mask=frame_mask, mode='norm')
    flat = flat.masked_fill(~frame_mask[:, None, :], 0)
    if not torch.isfinite(flat).all():
        raise ValueError('Nonfinite normalized input')
    patch_mask = frame_mask.unfold(-1, patch_len, patch_len).any(-1)
    # Expanded mask is only for patch embedding selection, NEVER for RevIN.
    patch_input_mask = patch_mask.repeat_interleave(patch_len, dim=-1)
    patches = model.tokenizer(x=flat)
    encoded = model.patch_embedding(patches, mask=patch_input_mask)
    encoded = encoded.squeeze(1)
    outputs = model.encoder(inputs_embeds=encoded, attention_mask=patch_mask.long())
    features = outputs.last_hidden_state.masked_fill(~patch_mask[..., None], 0)
    if not torch.isfinite(features).all():
        raise ValueError('Nonfinite MOMENT embeddings')
    count, dim = features.shape[-2:]
    features = features.reshape(b, channels, count, dim).permute(0, 2, 1, 3).contiguous()
    valid = masks.unfold(-1, patch_len, patch_len).any(-1)
    return features, valid[:, :, None].expand(b, count, channels)


def load_moment(path, device):
    # Kept lazy so preprocessing and protocol tests do not load TSFM dependencies.
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        str(path), model_kwargs={'task_name': 'embedding', 'reduction': 'none'},
        local_files_only=True)
    model.init()
    return model.to(device).eval().requires_grad_(False)


def embed_store(raw_path, output_path, model, device, provenance, batch_size=4, reduction='none'):
    raw_index, raw_provenance = store_index(raw_path)
    expected = embedding_provenance(provenance, raw_provenance, reduction)
    if Path(output_path).exists():
        index, actual = store_index(output_path)
        if actual != expected:
            raise ValueError(f'Embedding provenance mismatch: {output_path}')
        return index

    def records():
        env = lmdb.open(str(raw_path), readonly=True, lock=False, create=False)
        try:
            for start in range(0, len(raw_index), batch_size):
                with env.begin() as txn:
                    batch = [pickle.loads(txn.get(m['key'].encode()))
                             for m in raw_index[start:start + batch_size]]
                samples = np.stack([r['sample'] for r in batch])
                masks = np.stack([r['mask'] for r in batch])
                pad = (-samples.shape[-1]) % model.patch_len
                samples = np.pad(samples, ((0, 0), (0, 0), (0, pad)))
                masks = np.pad(masks, ((0, 0), (0, pad)))
                features, valid = masked_moment_embeddings(
                    model, torch.from_numpy(samples).to(device), torch.from_numpy(masks).to(device))
                features, valid = arrange_embeddings(features, valid, reduction)
                for row, x, mask in zip(batch, features.cpu().numpy(), valid.cpu().numpy()):
                    yield dict(sample=x.astype(np.float32), mask=mask, meta=row['meta'])
                done = min(start + batch_size, len(raw_index))
                if start == 0 or done == len(raw_index) or done // 100 != start // 100:
                    print(f'Embedding {output_path}: {done}/{len(raw_index)} samples', flush=True)
        finally:
            env.close()
    return write_store(output_path, records(), expected)


class EmbeddingStore:
    """One read-only LMDB handle shared by train/validation views (no key reset)."""
    def __init__(self, path):
        self.index, self.provenance = store_index(path)
        self.by_key = {m['key']: m for m in self.index}
        self.env = lmdb.open(str(path), readonly=True, lock=False, create=False)
        first = self.read(self.index[0]['key'])
        self.shape = tuple(first['sample'].shape)

    def read(self, key):
        with self.env.begin() as txn:
            value = txn.get(key.encode())
            if value is None:
                raise KeyError(key)
            return pickle.loads(value)

    def view(self, held_out_patient=None, role='train'):
        return EmbeddingView(self, fold_keys(self.index, held_out_patient, role))

    def close(self):
        self.env.close()


class EmbeddingView(Dataset):
    def __init__(self, store, keys):
        self.store, self.keys = store, tuple(keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        record = self.store.read(key)
        if record['sample'].shape != self.store.shape:
            raise ValueError(f'Embedding shape mismatch for {key}')
        return (torch.from_numpy(record['sample']), record['meta']['label'],
                torch.from_numpy(record['mask']), key)

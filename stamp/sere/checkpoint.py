"""Atomic epoch checkpoints and portable artifact references."""
import copy
import fcntl
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


@contextmanager
def run_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.run.lock').open('a+') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another process is running {directory}; do not start duplicate writers') from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def rng_state(generator, device):
    np_state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[np_state[0], np_state[1].tolist(),
                np_state[2], np_state[3], np_state[4]], cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None,
                loader=generator.get_state())


def restore_rng(state, generator, device):
    random.setstate(state['python'])
    name, values, pos, has_gauss, cached = state['numpy']
    np.random.set_state((name, np.asarray(values, dtype=np.uint32), pos, has_gauss, cached))
    torch.set_rng_state(state['cpu'])
    if torch.device(device).type == 'cuda' and state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)
    generator.set_state(state['loader'])


def portable_run(run, root):
    result = copy.deepcopy(run)
    result['checkpoint'] = str(Path(result['checkpoint']).resolve().relative_to(Path(root).resolve()))
    return result


def resolve_run(run, root):
    result = copy.deepcopy(run)
    result['checkpoint'] = str((Path(root) / result['checkpoint']).resolve())
    return result

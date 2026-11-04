"""Run independent fold/seed fits on one or more devices using spawned workers."""
import multiprocessing as mp
import torch

from stamp.sere.embedding import EmbeddingStore, EmbeddingView
from .training import run_stage

_DEVICE = None


def _initialize_worker(queue):
    global _DEVICE
    _DEVICE = queue.get()
    torch.set_num_threads(2)
    if torch.device(_DEVICE).type == 'cuda':
        torch.cuda.set_device(_DEVICE)


def stage_task(task, device=None):
    task = dict(task)
    tag = task.pop('tag')
    store = EmbeddingStore(task.pop('data_path'))
    validate = task.pop('validate')
    try:
        identity = task['identity']
        if not validate:
            raise ValueError('Inner validation required')
        train = EmbeddingView(store,[m['key'] for m in store.index if m['pid'] in identity['train_patients']])
        val = EmbeddingView(store,[m['key'] for m in store.index if m['pid'] in identity['held_out_patients']])
        result = run_stage(train_view=train, val_view=val, device=device or _DEVICE, **task)
        return tag, result
    finally:
        store.close()


def dispatch(tasks, devices):
    devices = [devices] if isinstance(devices, str) else list(devices)
    if not devices:
        raise ValueError('At least one training device is required')
    devices = [f'cuda:{torch.device(d).index or 0}' if torch.device(d).type == 'cuda' else d for d in devices]
    cuda_devices = [d for d in devices if torch.device(d).type == 'cuda']
    if len(set(cuda_devices)) != len(cuda_devices):
        raise ValueError('Use a different GPU for each worker')
    if cuda_devices and (not torch.cuda.is_available() or
                         any(torch.device(d).index >= torch.cuda.device_count() for d in cuda_devices)):
        raise ValueError(f'Requested devices {cuda_devices} are not all visible on this machine')
    if len(devices) == 1:
        return dict(stage_task(task, devices[0]) for task in tasks)
    context = mp.get_context('spawn')
    queue = context.Queue()
    for device in devices:
        queue.put(device)
    # Pool context terminates workers on Ctrl+C/errors; completed epoch files
    # survive. The next invocation can use different devices or worker counts.
    try:
        with context.Pool(len(devices), initializer=_initialize_worker, initargs=(queue,)) as pool:
            return dict(pool.imap_unordered(stage_task, tasks))
    finally:
        queue.close()

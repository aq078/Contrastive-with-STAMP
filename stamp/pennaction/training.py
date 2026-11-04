"""Five-seed, fixed-split STAMP training with validation-selected checkpoints."""
import copy
import inspect
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from stamp.common import setup_seed
from stamp.modeling.stamp import STAMP
from stamp.modeling.supcon_trainer import SupervisedContrastiveLoss
from stamp.sere.training import contrastive_loss, training_mode
from stamp.sere.checkpoint import atomic_torch_save, rng_state, restore_rng, run_lock
from stamp.sere.data import digest, file_digest, read_json, write_json
from .embedding import VideoDataset


def metrics(truths, predictions, n_classes):
    truth, pred = np.asarray(truths, dtype=int), np.asarray(predictions, dtype=int)
    if truth.shape != pred.shape or not len(truth) or ((truth < 0) | (truth >= n_classes)).any() or ((pred < 0) | (pred >= n_classes)).any():
        raise ValueError('Invalid multiclass predictions')
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (truth, pred), 1)
    support = cm.sum(1)
    if not (support > 0).all():
        raise ValueError('BA requires every configured action in this split')
    recalls = np.diag(cm) / support
    return dict(ba=float(recalls.mean()), accuracy=float((truth == pred).mean()),
                per_class_recall=recalls.tolist(), class_support=support.tolist(),
                predicted_class_counts=cm.sum(0).tolist(), confusion_matrix=cm.tolist(), instances=len(truth))


def build_model(params, shape, stage, n_classes):
    params = copy.deepcopy(params)
    t, s, d = shape
    params.update(input_dim=d, n_temporal_channels=t, n_spatial_channels=s, n_classes=n_classes)
    params['supcon_params'] = params.get('supcon_params', {}).get('proj_head') if stage == 'supcon' else None
    fields = inspect.signature(STAMP).parameters
    model = STAMP(**{k: v for k, v in params.items() if k in fields})
    if stage == 'head':
        model.requires_grad_(False)
        model.multi_head_attention_pooling.requires_grad_(True)
        model.classifier.requires_grad_(True)
    return model


def batches(data, batch_size, seed, shuffle=False):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate(model, data, stage, criterion, params, device, n_classes, save_predictions=False):
    model.eval()
    total, count, truth, pred, records = 0., 0, [], [], []
    for x, y, mask, videos in data:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        if stage == 'supcon':
            loss = contrastive_loss(model, x, y, mask, criterion, params)
        else:
            logits = model(x, valid_mask=mask)[0]
            loss = criterion(logits, y)
            truth.extend(y.cpu().tolist())
            guesses = logits.argmax(-1).cpu().tolist()
            pred.extend(guesses)
            if save_predictions:
                records.extend(dict(video=v, label=int(label), prediction=g, probabilities=p)
                               for v, label, g, p in zip(videos, y.cpu().tolist(), guesses,
                                                         logits.softmax(-1).cpu().tolist()))
        total += float(loss) * len(y)
        count += len(y)
    result = dict(loss=total / count)
    if stage != 'supcon':
        result.update(metrics(truth, pred, n_classes))
        if save_predictions:
            result['predictions'] = records
    return result


def model_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def fit_task(task, device):
    root, run_dir = Path(task['root']), Path(task['run_dir'])
    with run_lock(run_dir):
        return _fit_task(task, root, run_dir, device)


def _fit_task(task, root, run_dir, device):
    config, manifest = task['config'], task['manifest']
    stage, seed, reduction, protocol_hash = (task[k] for k in ('stage', 'seed', 'reduction', 'protocol_hash'))
    params = config['recipes'][{'baseline': 'baseline', 'supcon': 'supcon_stage1', 'head': 'supcon_stage2'}[stage]]
    initialization = task.get('initialization')
    request = dict(protocol_hash=protocol_hash, seed=seed, reduction=reduction, stage=stage,
                   params=params, initialization=initialization['checkpoint_sha256'] if initialization else None)
    signature = digest(request)
    if (run_dir / 'request.json').exists() and read_json(run_dir / 'request.json') != request:
        raise ValueError(f'Existing run differs: {run_dir}')
    write_json(run_dir / 'request.json', request)
    if (run_dir / 'run.json').exists():
        result = read_json(run_dir / 'run.json')
        if result['signature'] != signature or file_digest(run_dir / 'best.pt') != result['checkpoint_sha256']:
            raise ValueError('Completed run identity/checkpoint mismatch')
        return result
    train = VideoDataset(root, manifest, 'train', config, protocol_hash, reduction)
    val = VideoDataset(root, manifest, 'val', config, protocol_hash, reduction)
    n_classes = len(manifest['classes'])
    setup_seed(seed)
    torch.backends.cudnn.benchmark = False
    model = build_model(params, train.shape, stage, n_classes).to(device)
    if stage == 'head':
        if initialization is None:
            raise ValueError('SupCon stage 2 requires the same seed/reduction stage 1')
        path = root / initialization['checkpoint']
        if file_digest(path) != initialization['checkpoint_sha256']:
            raise ValueError('Stage 1 checkpoint changed')
        source = torch.load(path, map_location='cpu', weights_only=True)
        if any(source['request'][k] != request[k] for k in ('protocol_hash', 'seed', 'reduction')) or source['request']['stage'] != 'supcon':
            raise ValueError('Cross-seed/reduction SupCon initialization')
        model.load_state_dict({k: v for k, v in source['model'].items() if not k.startswith('proj_head.')}, strict=True)
    train_batches = batches(train, params['train_batch_size'], seed, True)
    # Video IDs are action-grouped. Use fixed shuffled batches for SupCon
    # validation so its loss is not evaluated on mostly single-action batches.
    val_batches = batches(val, params['test_batch_size'], seed, shuffle=stage == 'supcon')
    opt = params['optimizer_params']
    optimizer = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam}[opt['optimizer_name']](
        [p for p in model.parameters() if p.requires_grad], lr=params['lr_params']['initial_lr'],
        betas=tuple(opt.get('betas', (.9, .999))), eps=opt.get('eps', 1e-8), weight_decay=opt.get('weight_decay', 0.))
    horizon = params['n_epochs']
    lr = params['lr_params']
    if lr.get('use_scheduler') and lr['scheduler_type'] != 'one_cycle':
        raise ValueError('Unsupported scheduler')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr['max_lr'],
        total_steps=horizon * len(train_batches)) if lr.get('use_scheduler') else None
    criterion = (SupervisedContrastiveLoss(params.get('supcon_params', {}).get('temperature', .07))
                 if stage == 'supcon' else torch.nn.CrossEntropyLoss(label_smoothing=params.get('label_smoothing', 0.)))
    curve, best, start = [], None, 1
    resume = run_dir / 'resume.pt'
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['signature'] != signature:
            raise ValueError('Resume identity mismatch')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if scheduler:
            scheduler.load_state_dict(saved['scheduler'])
        curve, best = saved['curve'], saved['best']
        start = saved['epoch'] + 1
        restore_rng(saved['rng'], train_batches.generator, device)
        print(f'Resuming {stage} seed {seed} epoch {start}', flush=True)
    for epoch in range(start, horizon + 1):
        training_mode(model, stage)
        loss_sum, count = 0., 0
        for x, y, mask, _ in train_batches:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (contrastive_loss(model, x, y, mask, criterion, params) if stage == 'supcon'
                    else criterion(model(x, valid_mask=mask)[0], y))
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            if params.get('use_gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            if scheduler:
                scheduler.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)
        cuda = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
        with torch.random.fork_rng(devices=cuda):
            torch.manual_seed(seed + 10000)
            val_batches.generator.manual_seed(seed + 20000)
            validation = evaluate(model, val_batches, stage, criterion, params, device, n_classes)
        row = dict(epoch=epoch, train_loss=loss_sum/count, validation=validation)
        curve.append(row)
        score = -validation['loss'] if stage == 'supcon' else validation['ba']
        if best is None or score > best['score']:
            best = dict(epoch=epoch, score=score, validation=validation, model=model_state(model))
        atomic_torch_save(dict(signature=signature, epoch=epoch, model=model_state(model),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
            curve=curve, best=best, rng=rng_state(train_batches.generator, device)), resume)
        print(f'{reduction} {stage} seed={seed} epoch={epoch}/{horizon} validation={validation}', flush=True)
    atomic_torch_save(dict(signature=signature, request=request, model=best['model'],
                           epoch=best['epoch'], shape=list(train.shape)), run_dir / 'best.pt')
    result = dict(signature=signature, seed=seed, stage=stage, best_epoch=best['epoch'],
                  validation=best['validation'], curve=curve,
                  checkpoint=str((run_dir / 'best.pt').relative_to(root)),
                  checkpoint_sha256=file_digest(run_dir / 'best.pt'))
    write_json(run_dir / 'run.json', result)
    return result


def test_model(config, manifest, root, protocol_hash, reduction, run, device):
    path = Path(root) / run['checkpoint']
    if file_digest(path) != run['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['signature'] != run['signature'] or saved['request']['protocol_hash'] != protocol_hash:
        raise ValueError('Test model provenance mismatch')
    stage = run['stage']
    if stage not in ('baseline', 'head'):
        raise ValueError('Test requires a trained classifier')
    params = config['recipes']['baseline' if stage == 'baseline' else 'supcon_stage2']
    data = VideoDataset(root, manifest, 'test', config, protocol_hash, reduction)
    if list(data.shape) != saved['shape']:
        raise ValueError('Test embedding layout differs')
    model = build_model(params, data.shape, stage, len(manifest['classes'])).to(device)
    model.load_state_dict(saved['model'])
    return evaluate(model, batches(data, params['test_batch_size'], run['seed']), stage,
                    torch.nn.CrossEntropyLoss(), params, device, len(manifest['classes']), True)

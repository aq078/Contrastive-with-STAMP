"""Inner-validation early stopping; outer test patients never enter training."""
import inspect
from pathlib import Path
import numpy as np
import torch
from stamp.common import setup_seed
from stamp.modeling.supcon_trainer import SupervisedContrastiveLoss
from stamp.sere.training import (build_model, training_mode, checkpoint_load, initialize_from_supcon,
                                 loader, contrastive_loss, evaluate, training_class_balance)
from stamp.sere.embedding import EmbeddingView
from stamp.sere.metrics import CLASS_FIRST, patient_class_metrics
from stamp.sere.data import digest, file_digest, read_json, write_json
from stamp.sere.checkpoint import atomic_torch_save, rng_state, restore_rng, run_lock, resolve_run


def run_stage(*args, **kwargs):
    bound = inspect.signature(_run_stage).bind(*args, **kwargs)
    with run_lock(bound.arguments['output_dir']):
        return _run_stage(*args, **kwargs)


def _run_stage(params, stage, train_view, val_view, identity, output_dir, device,
              epochs=None, initialization=None):
    """Use only inner validation to stop; save and restore its best checkpoint."""
    if val_view is None:
        raise ValueError('Inner validation is required')
    outer = set(identity['outer_test_patients'])
    train_pids = {train_view.store.by_key[k]['pid'] for k in train_view.keys}
    val_pids = {val_view.store.by_key[k]['pid'] for k in val_view.keys}
    if outer & (train_pids | val_pids) or train_pids != set(identity['train_patients']):
        raise ValueError('Outer-test leakage or unexpected training patients')
    if train_view.store.provenance.get('data_scope') != 'full_trials':
        raise ValueError('Full-trial embeddings required')
    stopping = params['early_stopping']
    patience, min_delta = int(stopping['patience']), float(stopping['min_delta'])
    min_epochs = int(stopping['min_epochs'])
    if patience < 1 or min_delta < 0 or not 1 <= min_epochs <= params['n_epochs']:
        raise ValueError('Invalid early stopping settings')
    if stage not in ('baseline', 'supcon', 'head'):
        raise ValueError(stage)
    if train_view.store.provenance['portion'] != 'development':
        raise ValueError('Training may only use development embeddings')
    if train_view.store.provenance.get('reduction', 'none') != identity['reduction']:
        raise ValueError('Training embeddings use a different reduction mode')
    if val_view is not None and val_view.store.provenance['portion'] != 'development':
        raise ValueError('Validation may only use development embeddings')
    train_pids = {train_view.store.by_key[k]['pid'] for k in train_view.keys}
    held_out = set(identity['held_out_patients'])
    if identity['phase'] == 'cv' and held_out & train_pids:
        raise ValueError('Held-out patients appear in grouped-fold training/replay')
    if val_view is not None:
        val_pids = {val_view.store.by_key[k]['pid'] for k in val_view.keys}
        if train_pids & val_pids or val_pids != held_out:
            raise ValueError('Invalid grouped-fold patient separation')
    elif identity['phase'] == 'final':
        if set(train_view.keys) != {m['key'] for m in train_view.store.index}:
            raise ValueError('Final fit must use all development samples')
    horizon = int(params['n_epochs'])
    epochs = horizon if epochs is None else int(epochs)
    if not 1 <= epochs <= horizon:
        raise ValueError('Epoch count outside the CV schedule horizon')
    if val_view is not None and epochs != horizon:
        raise ValueError('LOPO curves must run the complete common epoch budget')
    source = None
    if initialization:
        source = dict(signature=initialization['signature'], epochs=initialization['epochs'],
                      sha256=file_digest(initialization['path']))
    request = dict(identity=identity, stage=stage, params=params, epochs=epochs,
                   train_keys_hash=digest(train_view.keys),
                   val_keys_hash=digest(val_view.keys) if val_view is not None else None,
                   data=train_view.store.provenance, initialization=source)
    balance = None
    if stage != 'supcon' and params.get('class_weighting') == 'balanced':
        balance = training_class_balance(train_view)
        request['class_balance'] = balance
    elif params.get('class_weighting') not in (None, 'none'):
        raise ValueError('class_weighting=balanced is supported only for baseline and head')
    signature = digest(request)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path, result_path = output_dir / 'request.json', output_dir / 'run.json'
    if request_path.exists() and read_json(request_path) != request:
        raise ValueError(f'Run directory belongs to a different configuration: {output_dir}')
    write_json(request_path, request)
    if result_path.exists():
        result = read_json(result_path)
        if result['signature'] != signature:
            raise ValueError('Completed run signature mismatch')
        if file_digest(output_dir / 'checkpoint.pt') != result['checkpoint_sha256']:
            raise ValueError('Completed checkpoint checksum mismatch')
        checkpoint_load(output_dir / 'checkpoint.pt', signature)
        print(f'Skipping completed run: {output_dir}', flush=True)
        return resolve_run(result, output_dir)

    seed = identity['seed']
    setup_seed(seed)
    torch.backends.cudnn.benchmark = False
    model = build_model(params, train_view.store.shape, stage).to(device)
    if stage == 'head':
        if initialization is None:
            raise ValueError('Stage 2 requires its fold/seed-specific SupCon checkpoint')
        initialize_from_supcon(model, initialization, identity, device)
    batches = loader(train_view, params['train_batch_size'], seed, shuffle=True)
    validation = {}
    if val_view is not None:
        for pid in sorted(held_out):
            view = EmbeddingView(val_view.store, [k for k in val_view.keys if val_view.store.by_key[k]['pid'] == pid])
            validation[str(pid)] = loader(view, params.get('test_batch_size', params['train_batch_size']), seed)
    opt = params['optimizer_params']
    optimizer_cls = {'adamw': torch.optim.AdamW, 'adam': torch.optim.Adam}[opt['optimizer_name']]
    optimizer = optimizer_cls(
        [p for p in model.parameters() if p.requires_grad], lr=params['lr_params']['initial_lr'],
        betas=tuple(opt.get('betas', (0.9, 0.999))), eps=opt.get('eps', 1e-8),
        weight_decay=opt.get('weight_decay', 0.0))
    scheduler = None
    lr = params['lr_params']
    if lr.get('use_scheduler'):
        if lr['scheduler_type'] != 'one_cycle':
            raise ValueError('SERE protocol currently supports the configured OneCycle recipes')
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr['max_lr'], total_steps=horizon * len(batches))
    criterion = (SupervisedContrastiveLoss(params.get('supcon_params', {}).get('temperature', 0.07))
                 if stage == 'supcon' else
                 torch.nn.CrossEntropyLoss(
                     weight=torch.tensor(balance['class_weights'], dtype=torch.float32, device=device) if balance else None,
                     label_smoothing=params.get('label_smoothing') or 0.0))
    curve = []
    best = None
    stale = 0
    resume_path = output_dir / 'resume.pt'
    start_epoch = 1
    if resume_path.exists():
        saved = checkpoint_load(resume_path, signature)
        if saved['epochs'] != epochs or saved['horizon'] != horizon:
            raise ValueError('Resume schedule does not match requested fit')
        model.load_state_dict(saved['model_state_dict'], strict=True)
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(saved['scheduler_state_dict'])
        curve = saved['curve']
        best, stale = saved['best'], saved['stale']
        start_epoch = saved['completed_epoch'] + 1
        if len(curve) != saved['completed_epoch'] or start_epoch > epochs + 1:
            raise ValueError('Invalid resume epoch history')
        restore_rng(saved['rng'], batches.generator, device)
        print(f'Resuming {output_dir} at epoch {start_epoch}/{epochs} on {device}', flush=True)
        del saved
    for epoch in range(start_epoch, epochs + 1):
        if stale >= patience and len(curve) >= min_epochs:
            break
        training_mode(model, stage)
        loss_sum, count = 0.0, 0
        for x, y, mask, _ in batches:
            x, y, mask = x.to(device), y.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (contrastive_loss(model, x, y, mask, criterion, params) if stage == 'supcon'
                    else criterion(model(x, valid_mask=mask)[0], y))
            if not torch.isfinite(loss):
                raise ValueError(f'Nonfinite training loss: {identity}, epoch {epoch}')
            loss.backward()
            if params.get('use_gradient_clipping'):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            loss_sum += float(loss.detach()) * len(y)
            count += len(y)
        row = dict(epoch=epoch, train_loss=loss_sum / count)
        if validation:
            devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
            with torch.random.fork_rng(devices=devices):
                # Validate patients separately (including SupCon batch context),
                # so large patients/folds cannot dominate the selection metric.
                row['patients'] = {}
                for pid, val_batches in validation.items():
                    torch.manual_seed(seed + 10000 + int(pid))
                    row['patients'][pid] = evaluate(model, val_batches, stage, criterion, params, device)
                row['loss'] = float(np.mean([r['loss'] for r in row['patients'].values()]))
                if stage != 'supcon':
                    if params.get('metric_aggregation') == CLASS_FIRST:
                        row.update(patient_class_metrics(row['patients']))
                    else:
                        row['ba'] = float(np.mean([r['ba'] for r in row['patients'].values()]))
        curve.append(row)
        score = -row['loss'] if stage == 'supcon' else row['ba']
        if best is None or score > best['score'] + min_delta:
            best = dict(epoch=epoch, score=float(score),
                        model={k: v.detach().cpu().clone() for k,v in model.state_dict().items()})
            stale = 0
        else:
            stale += 1
        # Commit only complete epochs. An interrupted batch/validation pass is
        # replayed from the previous epoch with optimizer/scheduler/RNG intact.
        atomic_torch_save(dict(signature=signature, completed_epoch=epoch, epochs=epochs,
                              horizon=horizon, curve=curve, best=best, stale=stale,
                              model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                              optimizer_state_dict=optimizer.state_dict(),
                              scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                              rng=rng_state(batches.generator, device)), resume_path)
        print(f"{identity['method']} {stage} L{identity['length']} "
              f"fold={identity['fold_id']} patients={identity['held_out_patients']} seed={seed} {row}", flush=True)
    checkpoint = dict(signature=signature, identity=identity, stage=stage, epochs=best['epoch'],
                      horizon=horizon, shape=list(train_view.store.shape),
                      model_state_dict=best['model'])
    atomic_torch_save(checkpoint, output_dir / 'checkpoint.pt')
    result = dict(signature=signature, identity=identity, stage=stage, epochs=best['epoch'],
                  trained_epochs=len(curve), early_stopped=len(curve) < epochs and stale >= patience and len(curve) >= min_epochs,
                  selection_metric='inner_validation_loss' if stage=='supcon' else 'inner_validation_ba',
                  checkpoint='checkpoint.pt',
                  checkpoint_sha256=file_digest(output_dir / 'checkpoint.pt'), curve=curve)
    write_json(result_path, result)
    return resolve_run(result, output_dir)

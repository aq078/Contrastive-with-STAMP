"""Five independent fold models per method/window; no validation or final refit."""
import copy
import csv
import tempfile
from pathlib import Path

import numpy as np
import torch

from stamp.sere.data import (create_manifest, digest, file_digest, iter_windows,
                             patient_folds, read_json, write_json, write_store, store_index)
from stamp.sere.embedding import EmbeddingStore, embed_store, embedding_provenance, load_moment
from stamp.sere.checkpoint import run_lock
from stamp.sere.metrics import classification_metrics, patient_class_metrics
from stamp.sere.training import build_model, checkpoint_load, loader
from stamp.sere.workers import dispatch

PROTOCOL = 'sere-full-trial-patient-5fold-evaluation-v1'


def full_trial_windows(config, manifest, length):
    # Reuse the tested window/missing-frame/normalization logic. The internal
    # 'development' store tag is required by the shared trainer, but its scope
    # is ALL original trial frames, not the previous 80% temporal partition.
    full = copy.deepcopy(manifest)
    for trial in full['trials']:
        trial['cut'] = trial['stop']
    yield from iter_windows(config['source_root'], full, 'development', length, config['stride'])


def initialize(config, reduction):
    if config['protocol'] != PROTOCOL or config['cv']['n_folds'] != 5:
        raise ValueError('Expected full-trial, five-fold evaluation protocol')
    if config['seeds'] != [654]:
        raise ValueError('This evaluation uses one fixed seed, 654, across five fold models')
    if reduction not in ('none', 'mean'):
        raise ValueError(reduction)
    if not config['window_lengths'] or len(set(config['window_lengths'])) != len(config['window_lengths']):
        raise ValueError('Distinct window lengths required')
    manifest = create_manifest(config['source_root'], config['exercises'])
    folds = [dict(fold_id=f['fold_id'], test_patients=f['validation_patients']) for f in
             patient_folds(manifest['patients'], config['cv']['n_folds'], config['cv']['seed'])]
    root = Path(config['output_dir']).resolve() / ('without_reduction' if reduction == 'none' else 'with_reduction')
    package = Path(__file__).resolve().parents[1]
    files = sorted((package/'sere_cv').glob('*.py')) + sorted((package/'sere').glob('*.py')) + sorted((package/'modeling').glob('*.py'))
    models = sorted(p for p in Path(config['moment_model_dir']).iterdir()
                    if p.is_file() and p.suffix in ('.json', '.bin', '.safetensors', '.pt', '.pth'))
    if not models:
        raise ValueError('Local MOMENT model files not found')
    snapshot = dict(protocol=PROTOCOL, reduction=reduction, manifest_hash=manifest['hash'], folds=folds,
        data_scope='full_trials', temporal_split=False, checkpoint_selection='fixed_last_epoch',
        config={k:v for k,v in config.items() if k not in ('source_root','moment_model_dir','output_dir','recipe_sources')},
        code_hash=digest({str(p.relative_to(package)):file_digest(p) for p in files}),
        model_hash=digest({p.name:file_digest(p) for p in models}))
    signature = digest(snapshot)
    snapshot['protocol_hash'] = signature
    with run_lock(root/'initialization'):
        path = root/'protocol.json'
        if path.exists() and read_json(path) != snapshot:
            raise ValueError('Incompatible saved protocol; use a new output directory')
        write_json(path, snapshot)
        write_json(root/'manifest.json', manifest)
        write_json(root/'folds.json', folds)
    return manifest, folds, root, signature


def cache_path(root, length):
    return Path(root)/'data'/f'L{length}'/'embeddings'


def provenance(manifest, signature, length, stride):
    return dict(protocol_hash=signature, manifest_hash=manifest['hash'], portion='development',
                data_scope='full_trials', temporal_split=False, length=length, stride=stride,
                aggregation='any', normalization='per_frame_hip_shoulder_then_div100',
                short_samples='pad_each_contiguous_run', schema=1)


def prepare(config, manifest, root, signature, reduction, device):
    moment = None
    for length in config['window_lengths']:
        parent = Path(root)/'data'/f'L{length}'
        parent.mkdir(parents=True, exist_ok=True)
        raw, target = parent/'raw', cache_path(root,length)
        prov = provenance(manifest, signature, length, config['stride'])
        if raw.exists():
            index, actual = store_index(raw)
            if actual != prov:
                raise ValueError('Raw full-trial cache provenance mismatch')
        else:
            with tempfile.TemporaryDirectory(prefix='raw-building-', dir=parent) as temp:
                staging = Path(temp)/'store'
                index = write_store(staging, full_trial_windows(config,manifest,length), prov)
                staging.rename(raw)
        if {m['pid'] for m in index} != set(manifest['patients']):
            raise ValueError('Missing patients in full-trial samples')
        write_json(parent/'coverage.json', dict(samples=len(index), patients=len(manifest['patients']),
                    padded_samples=sum(m['valid_length']<length for m in index),data_scope='full_trials'))
        expected = embedding_provenance(prov,prov,reduction)
        if target.exists():
            _, actual = store_index(target)
            if actual != expected:
                raise ValueError('Embedding full-trial cache provenance mismatch')
        else:
            if moment is None:
                print(f'Loading local MOMENT on {device}',flush=True)
                moment = load_moment(config['moment_model_dir'],device)
            with tempfile.TemporaryDirectory(prefix='embedding-building-',dir=parent) as temp:
                staging = Path(temp)/'store'
                embed_store(raw,staging,moment,device,prov,config.get('embedding_batch_size',4),reduction)
                staging.rename(target)
        print(f'Prepared full-trial {reduction} L{length}: {len(index)} windows',flush=True)
    del moment
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def tasks_for(config, folds, root, signature, reduction, length, stage, pretrained=None):
    recipe = config['recipes'][{'baseline':'baseline','supcon':'supcon_stage1','head':'supcon_stage2'}[stage]]
    tasks = []
    for fold in folds:
        fid = fold['fold_id']
        tag = str(fid)
        identity = dict(protocol_hash=signature, method='baseline' if stage=='baseline' else 'supcon',
                        length=length, fold_id=fid, held_out_patients=sorted(fold['test_patients']),
                        seed=654, phase='cv', reduction=reduction, evaluation_protocol=PROTOCOL)
        source = pretrained[tag] if pretrained else None
        tasks.append(dict(tag=tag,params=recipe,stage=stage,data_path=str(cache_path(root,length)),
            validate=False,identity=identity,output_dir=str(Path(root)/'runs'/f'L{length}'/stage/f'fold_{fid}'/'seed_654'),
            epochs=recipe['n_epochs'],initialization=dict(path=source['checkpoint'],signature=source['signature'],
            epochs=source['epochs']) if source else None))
    return tasks


def train(config,manifest,folds,root,signature,reduction,devices):
    for length in config['window_lengths']:
        _, actual = store_index(cache_path(root,length))
        if (actual.get('data_scope')!='full_trials' or actual['protocol_hash']!=signature
                or actual['manifest_hash']!=manifest['hash'] or actual['reduction']!=reduction):
            raise ValueError('Prepare this full-trial protocol before training')
        dispatch(tasks_for(config,folds,root,signature,reduction,length,'baseline'),devices)
        pretrained = dispatch(tasks_for(config,folds,root,signature,reduction,length,'supcon'),devices)
        dispatch(tasks_for(config,folds,root,signature,reduction,length,'head',pretrained),devices)


@torch.no_grad()
def evaluate_fold(config,manifest,fold,root,signature,reduction,length,method,device):
    stage = 'baseline' if method=='baseline' else 'head'
    directory = Path(root)/'runs'/f'L{length}'/stage/f"fold_{fold['fold_id']}"/'seed_654'
    run = read_json(directory/'run.json')
    ident = run['identity']
    expected = dict(protocol_hash=signature,method=method,length=length,fold_id=fold['fold_id'],
        held_out_patients=sorted(fold['test_patients']),seed=654,phase='cv',reduction=reduction,evaluation_protocol=PROTOCOL)
    params = config['recipes']['baseline' if method=='baseline' else 'supcon_stage2']
    if ident!=expected or run['epochs']!=params['n_epochs']:
        raise ValueError('Fold checkpoint provenance or fixed duration differs')
    checkpoint = directory/'checkpoint.pt'
    if file_digest(checkpoint)!=run['checkpoint_sha256']:
        raise ValueError('Fold checkpoint checksum mismatch')
    path = Path(root)/'test'/f'L{length}'/method/f"fold_{fold['fold_id']}.json"
    if path.exists():
        saved = read_json(path)
        if saved['signature']!=run['signature'] or saved['checkpoint_sha256']!=run['checkpoint_sha256']:
            raise ValueError('Saved fold predictions refer to another model')
        return saved
    store = EmbeddingStore(cache_path(root,length))
    try:
        if store.provenance['protocol_hash']!=signature or store.provenance.get('data_scope')!='full_trials':
            raise ValueError('Wrong full-trial evaluation cache')
        view = store.view(fold['test_patients'],'val')  # indexing only; never used during training
        model = build_model(params,store.shape,stage).to(device)
        payload = checkpoint_load(checkpoint,run['signature'])
        model.load_state_dict(payload['model_state_dict'])
        model.eval()
        records = []
        for x,y,mask,keys in loader(view,params['test_batch_size'],654):
            probabilities = model(x.to(device),valid_mask=mask.to(device))[0].softmax(-1).cpu().tolist()
            records.extend(dict(key=k,pid=store.by_key[k]['pid'],label=int(label),pred=int(np.argmax(prob)),probabilities=prob)
                           for k,label,prob in zip(keys,y.tolist(),probabilities))
        patients = {}
        for pid in fold['test_patients']:
            rows = [r for r in records if r['pid']==pid]
            patients[str(pid)] = classification_metrics([r['label'] for r in rows],[r['pred'] for r in rows])
        result = dict(signature=run['signature'],checkpoint_sha256=run['checkpoint_sha256'],fold_id=fold['fold_id'],
                      test_patients=fold['test_patients'],patients=patients,predictions=records,
                      metrics=patient_class_metrics(patients))
        write_json(path,result)
        return result
    finally:
        store.close()


def summarize_folds(results,expected_patients):
    patients,seen_keys = {},set()
    for result in results:
        if set(patients)&set(result['patients']):
            raise ValueError('Patient evaluated in multiple test folds')
        patients.update(result['patients'])
        for record in result['predictions']:
            if record['key'] in seen_keys:
                raise ValueError('Window evaluated in multiple test folds')
            seen_keys.add(record['key'])
    if set(patients)!={str(p) for p in expected_patients}:
        raise ValueError('Incomplete test-fold patient coverage')
    scores=[r['metrics']['ba'] for r in results]
    pooled=[r['metrics']['pooled_ba'] for r in results]
    out_of_fold=patient_class_metrics(patients)
    return dict(test_ba=float(np.mean(scores)),test_fold_std_ba=float(np.std(scores,ddof=1)),
                test_pooled_ba=out_of_fold['pooled_ba'],test_fold_mean_pooled_ba=float(np.mean(pooled)),
                test_fold_std_pooled_ba=float(np.std(pooled,ddof=1)),
                fold_scores={str(r['fold_id']):r['metrics']['ba'] for r in results},
                out_of_fold_metrics=out_of_fold,folds=len(results),std_ddof=1)


def test(config,manifest,folds,root,signature,reduction,device):
    rows=[]
    for length in config['window_lengths']:
        for method in ('baseline','supcon'):
            results=[evaluate_fold(config,manifest,f,root,signature,reduction,length,method,device) for f in folds]
            measured=summarize_folds(results,manifest['patients'])
            rows.append(dict(method=method,reduction=reduction,window_length=length,seed=654,**measured))
    report=dict(protocol_hash=signature,protocol=PROTOCOL,aggregation='equal_test_folds',
                hyperparameter_selection=False,final_retraining=False,rows=rows)
    write_json(Path(root)/'evaluation_results.json',report)
    flat=[{k:v for k,v in r.items() if not isinstance(v,dict)} for r in rows]
    target=Path(root)/'evaluation_results.csv'
    temporary=target.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    temporary.replace(target)
    return report

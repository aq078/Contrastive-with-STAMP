"""Choose inner validation using outer-training patients only, once for all lengths."""
import itertools
from pathlib import Path
import random

from stamp.sere.data import digest, read_json, write_json, store_index


def inner_folds(config,manifest,outer_folds,root,signature):
    # Never inspect an outer-test patient's label counts while choosing that
    # fold's inner partition. Same seeded partition for every method/reduction.
    n = config['inner_validation']['patients']
    indices = {length: store_index(Path(root)/'data'/f'L{length}'/'embeddings')[0]
               for length in config['window_lengths']}
    result=[]
    for fold in outer_folds:
        eligible = sorted(set(manifest['patients'])-set(fold['test_patients']))
        if not 1 <= n < len(eligible):
            raise ValueError('Invalid inner-validation patient count')
        candidates=list(itertools.combinations(eligible,n))
        random.Random(config['inner_validation']['seed']+fold['fold_id']).shuffle(candidates)
        supports={length:{p:{m['label'] for m in index if m['pid']==p} for p in eligible}
                  for length,index in indices.items()}
        for candidate in candidates:
            training=set(eligible)-set(candidate)
            if all(set.union(*(supports[length][p] for p in candidate))=={0,1}
                   and set.union(*(supports[length][p] for p in training))=={0,1} for length in indices):
                result.append(dict(fold_id=fold['fold_id'],test_patients=fold['test_patients'],
                                   validation_patients=list(candidate),train_patients=sorted(training)))
                break
        else:
            raise ValueError(f'No inner split with both classes for fold {fold["fold_id"]}')
    payload=dict(protocol_hash=signature,folds=result,policy=config['inner_validation'])
    payload['hash']=digest(payload)
    path=Path(root)/'inner_folds.json'
    if path.exists() and read_json(path)!=payload:
        raise ValueError('Inner split changed; do not overwrite existing runs')
    write_json(path,payload)
    return result

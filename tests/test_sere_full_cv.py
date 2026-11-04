import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from stamp.sere.data import create_manifest,read_json,write_store
from stamp.sere.embedding import embedding_provenance
from stamp.sere_cv.workflow import (PROTOCOL,full_trial_windows,cache_path,provenance,
                                    tasks_for,train,test as evaluate_cv,summarize_folds)
from test_sere_protocol import small_params,TemporalDataTests


class FullCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_windows_use_whole_trial_including_old_test_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            TemporalDataTests().make_source(tmp)
            manifest=create_manifest(tmp,['E1','E3'])
            original=copy.deepcopy(manifest)
            windows=list(full_trial_windows(dict(source_root=tmp,stride=4),manifest,32))
            self.assertEqual(manifest,original)
            by_id={t['trial_id']:t for t in manifest['trials']}
            self.assertTrue(any(w['meta']['frame_start']<by_id[w['meta']['trial_id']]['cut']
                                <w['meta']['frame_stop'] for w in windows))
            self.assertTrue(all(w['meta']['frame_stop']<=by_id[w['meta']['trial_id']]['stop'] for w in windows))

    def test_end_to_end_five_models_no_validation_no_refit_and_resume(self):
        for reduction in ['none','mean']:
            with self.subTest(reduction=reduction),tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                root=Path(tmp)
                config=dict(seeds=[654],stride=4,window_lengths=[8,16],
                            recipes={k:copy.deepcopy(small_params()) for k in ['baseline','supcon_stage1','supcon_stage2']})
                for key in ['baseline','supcon_stage2']:
                    config['recipes'][key]['class_weighting']='balanced'
                manifest=dict(patients=[1,2,3,4,5],hash='manifest')
                folds=[dict(fold_id=p,test_patients=[p]) for p in manifest['patients']]
                rng=np.random.default_rng(3)
                for length in config['window_lengths']:
                    prov=provenance(manifest,'protocol',length,4)
                    records=[dict(sample=rng.normal(size=(2,3,6)).astype('float32'),mask=np.ones((2,3),dtype=bool),
                                  meta=dict(key=f'p{p}_{i}',pid=p,label=i%2,portion='development'))
                             for p in manifest['patients'] for i in range(4)]
                    write_store(cache_path(root,length),records,embedding_provenance(prov,prov,reduction))
                tasks=tasks_for(config,folds,root,'protocol',reduction,8,'baseline')
                self.assertEqual(len(tasks),5)
                self.assertTrue(all(t['validate'] is False for t in tasks))
                train(config,manifest,folds,root,'protocol',reduction,['cpu'])
                self.assertFalse((root/'test').exists())
                checksums={p:p.read_bytes() for p in root.glob('runs/**/request.json')}
                for path in checksums:
                    request=read_json(path)
                    self.assertIsNone(request['val_keys_hash'])
                    run=read_json(path.parent/'run.json')
                    self.assertEqual(run['epochs'],3)
                    self.assertTrue(all('patients' not in row for row in run['curve']))
                    if request['stage']!='supcon':
                        self.assertEqual(request['class_balance']['class_counts'],[8,8])
                result=evaluate_cv(config,manifest,folds,root,'protocol',reduction,'cpu')
                self.assertEqual(len(result['rows']),4)
                self.assertFalse(result['hyperparameter_selection'])
                self.assertFalse(result['final_retraining'])
                for row in result['rows']:
                    self.assertEqual(row['folds'],5)
                    self.assertAlmostEqual(row['test_ba'],np.mean(list(row['fold_scores'].values())))
                    self.assertAlmostEqual(row['test_fold_std_ba'],np.std(list(row['fold_scores'].values()),ddof=1))
                for p in (root/'test').glob('L*/*/fold_*.json'):
                    fold=read_json(p)
                    self.assertEqual({r['pid'] for r in fold['predictions']},set(fold['test_patients']))
                train(config,manifest,folds,root,'protocol',reduction,['cpu'])
                self.assertEqual(result,evaluate_cv(config,manifest,folds,root,'protocol',reduction,'cpu'))
                self.assertEqual(checksums,{p:p.read_bytes() for p in checksums})

    def test_temporal_cache_cannot_be_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            config=dict(window_lengths=[8])
            write_store(cache_path(root,8),[dict(sample=np.zeros((2,3,6)),mask=np.ones((2,3)),
                        meta=dict(key='x',pid=1,label=0,portion='development'))],
                        dict(protocol_hash='old',portion='development'))
            with self.assertRaisesRegex(ValueError,'full-trial protocol'):
                train(config,{'hash':'m'},[],root,'new','none',['cpu'])

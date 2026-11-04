import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stamp.sere.data import read_json, write_store
from stamp.sere.embedding import EmbeddingStore, EmbeddingView, embedding_provenance
from stamp.sere.checkpoint import atomic_torch_save
from stamp.sere_early.splits import inner_folds
from stamp.sere_early.workflow import (provenance,cache_path,tasks_for,train,test as evaluate_cv)
from stamp.sere_early.training import run_stage
from test_sere_protocol import small_params


class EarlyCVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self,root,reduction='none'):
        config=dict(window_lengths=[8],seeds=[654],stride=4,
                    inner_validation=dict(patients=1,seed=42),recipes={})
        for stage in ['baseline','supcon_stage1','supcon_stage2']:
            params=copy.deepcopy(small_params())
            params.update(n_epochs=6,early_stopping=dict(patience=2,min_epochs=1,min_delta=0.0))
            if stage!='supcon_stage1':
                params.update(class_weighting='balanced',metric_aggregation='class_first_patient_recall')
            config['recipes'][stage]=params
        manifest=dict(patients=list(range(1,7)),hash='manifest')
        folds=[dict(fold_id=i,test_patients=([1,6] if i==1 else [i])) for i in range(1,6)]
        rng=np.random.default_rng(0)
        records=[dict(sample=rng.normal(size=(2,3,6)).astype('float32'),mask=np.ones((2,3),dtype=bool),
                      meta=dict(key=f'{p}_{i}',pid=p,label=i%2,portion='development'))
                 for p in manifest['patients'] for i in range(4)]
        prov=provenance(manifest,'protocol',8,4)
        write_store(cache_path(root,8),records,embedding_provenance(prov,prov,reduction))
        return config,manifest,folds

    @staticmethod
    def validation(*args,**kwargs):
        if args[2]=='supcon':return {'loss':1.0}
        return dict(loss=1.0,ba=.5,confusion_matrix=[[0,2],[0,2]])

    def test_five_fold_both_modes_early_stop_best_checkpoint_and_resume(self):
        for reduction in ['none','mean']:
            with self.subTest(reduction=reduction),tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                root=Path(tmp)
                config,manifest,folds=self.fixture(root,reduction)
                inner=inner_folds(config,manifest,folds,root,'protocol')
                for f in inner:
                    tr,va,te=map(set,[f['train_patients'],f['validation_patients'],f['test_patients']])
                    self.assertFalse(tr&va or tr&te or va&te)
                    self.assertEqual(tr|va|te,set(manifest['patients']))
                with patch('stamp.sere_early.training.evaluate',side_effect=self.validation):
                    train(config,manifest,folds,root,'protocol',reduction,['cpu'])
                self.assertFalse((root/'test').exists())
                for path in root.glob('runs/**/run.json'):
                    run=read_json(path)
                    self.assertEqual(run['trained_epochs'],3)
                    self.assertEqual(run['epochs'],1)
                    self.assertTrue(run['early_stopped'])
                    request=read_json(path.parent/'request.json')
                    if run['stage']!='supcon':
                        count=len(run['identity']['train_patients'])*2
                        self.assertEqual(request['class_balance']['class_counts'],[count,count])
                    if run['stage']=='head':
                        self.assertEqual(request['initialization']['epochs'],1)
                    saved=torch.load(path.parent/'checkpoint.pt',weights_only=True)
                    resume=torch.load(path.parent/'resume.pt',weights_only=True)
                    self.assertTrue(all(torch.equal(v,resume['best']['model'][k]) for k,v in saved['model_state_dict'].items()))
                result=evaluate_cv(config,manifest,folds,root,'protocol',reduction,'cpu')
                self.assertEqual(len(result['rows']),2)
                self.assertEqual(result['epoch_selection'],'inner_validation')
                for p in (root/'test').glob('L*/*/*.json'):
                    r=read_json(p)
                    self.assertEqual({x['pid'] for x in r['predictions']},set(r['test_patients']))
                with patch('stamp.sere_early.training.evaluate') as forbidden:
                    train(config,manifest,folds,root,'protocol',reduction,['cpu'])
                    forbidden.assert_not_called()
                self.assertEqual(result,evaluate_cv(config,manifest,folds,root,'protocol',reduction,'cpu'))

    def test_minimum_epochs_prevent_stopping_during_lr_ramp(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            config,manifest,folds=self.fixture(root)
            config['recipes']['baseline']['early_stopping']['min_epochs']=5
            inner=inner_folds(config,manifest,folds,root,'protocol')
            task=tasks_for(config,inner,root,'protocol','none',8,'baseline')[0]
            from stamp.sere_early.workers import stage_task
            with patch('stamp.sere_early.training.evaluate',side_effect=self.validation):
                _,result=stage_task(task,'cpu')
            self.assertEqual(result['trained_epochs'],5)
            self.assertEqual(result['epochs'],1)

    def test_outer_test_labels_cannot_change_inner_split(self):
        config=dict(window_lengths=[8,16],inner_validation=dict(patients=2,seed=42))
        manifest=dict(patients=[1,2,3,4,5,6])
        folds=[dict(fold_id=1,test_patients=[1,2])]
        index=[dict(pid=p,label=y) for p in manifest['patients'] for y in [0,1]]
        changed=[dict(r,label=1) if r['pid'] in [1,2] else r for r in index]
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            with patch('stamp.sere_early.splits.store_index',return_value=(index,{})):
                first=inner_folds(config,manifest,folds,Path(a),'protocol')
            with patch('stamp.sere_early.splits.store_index',return_value=(changed,{})):
                second=inner_folds(config,manifest,folds,Path(b),'protocol')
            self.assertEqual(first,second)

    def test_resume_preserves_patience_and_best_state_and_rejects_test_leakage(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            config,manifest,folds=self.fixture(root)
            inner=inner_folds(config,manifest,folds,root,'protocol')
            task=tasks_for(config,inner,root,'protocol','none',8,'baseline')[0]
            ident=task['identity']
            store=EmbeddingStore(cache_path(root,8))
            try:
                tr=EmbeddingView(store,[m['key'] for m in store.index if m['pid'] in ident['train_patients']])
                va=EmbeddingView(store,[m['key'] for m in store.index if m['pid'] in ident['held_out_patients']])
                kwargs=dict(params=task['params'],stage='baseline',train_view=tr,val_view=va,identity=ident,device='cpu')
                with patch('stamp.sere_early.training.evaluate',side_effect=self.validation):
                    ref=run_stage(output_dir=root/'reference',**kwargs)
                    def interrupt(payload,path):
                        atomic_torch_save(payload,path)
                        if path.name=='resume.pt' and payload['completed_epoch']==2:raise KeyboardInterrupt()
                    with patch('stamp.sere_early.training.atomic_torch_save',side_effect=interrupt):
                        with self.assertRaises(KeyboardInterrupt):run_stage(output_dir=root/'interrupted',**kwargs)
                    resumed=run_stage(output_dir=root/'interrupted',**kwargs)
                self.assertEqual(ref['curve'],resumed['curve'])
                self.assertEqual(resumed['trained_epochs'],3)
                self.assertEqual(resumed['epochs'],1)
                bad=copy.deepcopy(ident);bad['outer_test_patients'].append(bad['train_patients'][0])
                with self.assertRaisesRegex(ValueError,'Outer-test leakage'):
                    run_stage(output_dir=root/'bad',**dict(kwargs,identity=bad))
            finally:
                store.close()

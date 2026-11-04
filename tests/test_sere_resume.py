"""Interruption, relocation, grouped-patient weighting and spawned-worker checks."""
import contextlib
import copy
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from test_sere_protocol import small_params
import test_sere_protocol as protocol_tests
from stamp.sere.checkpoint import atomic_torch_save, run_lock, resolve_run, portable_run
from stamp.sere.data import patient_folds, write_store, write_json, digest
from stamp.sere.embedding import EmbeddingStore, embedding_provenance
from stamp.sere.metrics import average_fold_curves
from stamp.sere.training import run_stage
from stamp.sere.workflow import identity, initialization, reused_development, initialize
from stamp.sere.workers import dispatch


class GroupedTests(unittest.TestCase):
    def test_five_folds_cover_all_18_patients_without_overlap(self):
        patients = list(range(1, 19))
        folds = patient_folds(patients)
        self.assertEqual([len(f['validation_patients']) for f in folds], [4, 4, 4, 3, 3])
        self.assertEqual(sorted(p for f in folds for p in f['validation_patients']), patients)
        self.assertEqual(patient_folds(list(reversed(patients))), folds)
        for fold in folds:
            self.assertFalse(set(fold['training_patients']) & set(fold['validation_patients']))
        # A fold-average would give 3/5; equal patient weighting gives 12/18.
        runs = {}
        for fold in folds:
            score = float(len(fold['validation_patients']) == 4)
            row = {'patients': {str(p): {'ba': score} for p in fold['validation_patients']}}
            runs[str(fold['fold_id'])] = {str(s): {'curve': [row]} for s in [1, 2]}
        result = average_fold_curves(runs, folds, patients, [1, 2], 'ba')
        self.assertAlmostEqual(result['score'], 2 / 3)

    def test_cache_reuse_is_read_only_and_checks_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'old' / 'without_reduction'
            new = root / 'new' / 'without_reduction'
            write_json(old / 'protocol.json', dict(protocol_hash='old', model_hash='weights', manifest_hash='frames'))
            write_json(new / 'protocol.json', dict(protocol_hash='new', model_hash='weights', manifest_hash='frames'))
            raw = dict(protocol_hash='old', manifest_hash='frames', portion='development', length=8,
                       stride=16, aggregation='any', normalization='per_frame_hip_shoulder_then_div100',
                       short_samples='pad_each_contiguous_run', schema=1)
            path = old / 'data' / 'L8' / 'development' / 'embeddings'
            write_store(path, [dict(sample=np.ones((1, 99, 6), dtype='float32'), mask=np.ones((1, 99), dtype=bool),
                                    meta=dict(key='1', pid=1, portion='development', label=0))],
                        embedding_provenance(raw, raw, 'none'))
            before = (path / 'data.mdb').stat().st_mtime_ns
            config = dict(prepared_data_dir=str(root / 'old'), reduction='none', stride=16)
            self.assertEqual(reused_development(config, {'hash': 'frames', 'patients': [1]}, new, 8), path)
            self.assertEqual((path / 'data.mdb').stat().st_mtime_ns, before)
            with self.assertRaises(ValueError):
                reused_development(dict(config, stride=8), {'hash': 'frames', 'patients': [1]}, new, 8)

    def test_protocol_can_move_to_another_mount_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol_tests.TemporalDataTests().make_source(root / 'source')
            (root / 'moment').mkdir()
            (root / 'moment' / 'config.json').write_text('{"patch_len": 8}')
            config = dict(protocol='sere-temporal80-lopo-v1', seeds=[654, 114, 25, 759, 281],
                          window_lengths=[8], stride=4, exercises=['E1', 'E3'], cv={'n_folds': 2, 'seed': 42},
                          source_root=str(root / 'source'), moment_model_dir=str(root / 'moment'),
                          output_dir=str(root / 'runs'), recipes={
                              k: small_params() for k in ['baseline', 'supcon_stage1', 'supcon_stage2']})
            path = root / 'config.json'
            write_json(path, config)
            original = initialize(path)[3]
            for name in ['source', 'moment', 'runs']:
                shutil.move(str(root / name), str(root / (name + '_moved')))
            config.update(source_root=str(root / 'source_moved'), moment_model_dir=str(root / 'moment_moved'),
                          output_dir=str(root / 'runs_moved'))
            write_json(path, config)
            self.assertEqual(initialize(path)[3], original)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make_data(self, root):
        path = root / 'data'
        rng = np.random.default_rng(3)
        records = []
        for pid in [1, 2, 3, 4]:
            for i in range(4):
                records.append(dict(sample=rng.normal(size=(2, 3, 6)).astype('float32'),
                                    mask=np.ones((2, 3), dtype=bool),
                                    meta=dict(key=f'{pid}_{i}', pid=pid, label=i % 2, portion='development')))
        write_store(path, records, {'portion': 'development', 'reduction': 'none'})
        return path

    def test_interrupted_cv_and_final_resume_exactly_for_all_stages_after_move(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            store = EmbeddingStore(self.make_data(root))
            params = small_params()
            try:
                for phase in ['cv', 'final']:
                    patients = [1, 2] if phase == 'cv' else None
                    train = store.view(patients, 'train') if patients else store.view(role='final')
                    val = store.view(patients, 'val') if patients else None
                    ident = identity('protocol', 'supcon', 16, patients, 654, phase, fold_id=1 if patients else None)
                    source = run_stage(params, 'supcon', train, None, ident, root / f'{phase}_source', 'cpu')
                    for stage in ['baseline', 'supcon', 'head']:
                        with self.subTest(phase=phase, stage=stage):
                            init = initialization(source) if stage == 'head' else None
                            control = run_stage(params, stage, train, val, ident,
                                                root / f'{phase}_{stage}_control', 'cpu', initialization=init)
                            directory = root / f'{phase}_{stage}_interrupted'
                            def interrupt_after_first_epoch(payload, path):
                                atomic_torch_save(payload, path)
                                if Path(path).name == 'resume.pt' and payload['completed_epoch'] == 1:
                                    raise KeyboardInterrupt('Simulated SSH disconnect')
                            with patch('stamp.sere.training.atomic_torch_save', side_effect=interrupt_after_first_epoch):
                                with self.assertRaises(KeyboardInterrupt):
                                    run_stage(params, stage, train, val, ident, directory, 'cpu', initialization=init)
                            self.assertFalse((directory / 'run.json').exists())
                            moved = root / f'{phase}_{stage}_new_mount'
                            shutil.move(str(directory), str(moved))
                            resumed = run_stage(params, stage, train, val, ident, moved, 'cpu', initialization=init)
                            self.assertEqual(control['curve'], resumed['curve'])
                            a = torch.load(control['checkpoint'], weights_only=True)['model_state_dict']
                            b = torch.load(resumed['checkpoint'], weights_only=True)['model_state_dict']
                            for key in a:
                                torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                            persisted = json.loads((moved / 'run.json').read_text())
                            self.assertEqual(persisted['checkpoint'], 'checkpoint.pt')
                            portable = portable_run(resumed, root)
                            self.assertFalse(Path(portable['checkpoint']).is_absolute())
                            self.assertEqual(resolve_run(portable, root)['checkpoint'], resumed['checkpoint'])
                            # Completed runs skip, even after relocation.
                            self.assertEqual(run_stage(params, stage, train, val, ident, moved, 'cpu',
                                                       initialization=init), resumed)
            finally:
                store.close()

    def test_duplicate_writers_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with run_lock(tmp):
                with self.assertRaises(RuntimeError):
                    with run_lock(tmp):
                        pass

    def test_spawned_workers_run_distinct_jobs_and_resume_with_one_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self.make_data(root)
            tasks = []
            for seed in [654, 114, 25]:
                tasks.append(dict(tag=str(seed), params=small_params(), stage='baseline', data_path=str(path),
                                  validate=True, identity=identity('protocol', 'baseline', 16, [1, 2], seed, fold_id=1),
                                  output_dir=str(root / f'run_{seed}'), epochs=1, initialization=None))
                tasks[-1]['params']['n_epochs'] = 1
            results = dispatch(tasks, ['cpu', 'cpu'])
            self.assertEqual(set(results), {'654', '114', '25'})
            for run in results.values():
                self.assertEqual(set(run['curve'][0]['patients']), {'1', '2'})
            resumed = dispatch(tasks, 'cpu')
            self.assertEqual(results, resumed)


if __name__ == '__main__':
    unittest.main()

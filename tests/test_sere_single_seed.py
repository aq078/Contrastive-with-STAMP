import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.sere_selection_policy import apply_selection_policy
from scripts.sere_single_seed import aggregate, compatible_stage_paths, select_single_seed
from scripts.sere_window_evaluation import fit_all_windows, test_all_windows as evaluate_all
from stamp.sere.data import read_json, write_json, write_store
from stamp.sere.embedding import embedding_provenance
from test_sere_protocol import small_params


class SingleSeedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sample_sd_uses_fold_scores_at_common_epoch_and_mean_stays_patient_equal(self):
        folds = [dict(fold_id=1, validation_patients=[1, 2, 3]),
                 dict(fold_id=2, validation_patients=[4, 5])]
        runs = {}
        for fold, scores in zip(folds, [[.9, .4], [.1, .5]]):
            runs[f"{fold['fold_id']}/654"] = dict(curve=[
                dict(patients={str(p): {'ba': value} for p in fold['validation_patients']})
                for value in scores])
        result = aggregate(runs, folds, [1, 2, 3, 4, 5], [654], 'ba')
        self.assertEqual(result['epoch'], 1)
        self.assertAlmostEqual(result['score'], .58)
        self.assertAlmostEqual(result['fold_mean_ba'], .5)
        self.assertAlmostEqual(result['fold_std_ba'], np.std([.9, .1], ddof=1))

    def test_one_seed_amendment_keeps_old_policy_and_other_seed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(window_lengths=[32, 64, 80], seeds=[654, 114, 25, 759, 281])
            path = root / 'config.json'
            write_json(path.with_suffix('.selection.json'), {'window_lengths': [32, 64, 80]})
            apply_selection_policy(config, path, root, 'protocol')
            before = (root / 'selection_policy.json').read_bytes()
            other = root / 'cv/L32/baseline/fold_1/seed_114/run.json'
            write_json(other, {'preserved': True})
            write_json(path.with_suffix('.selection.json'),
                       dict(window_lengths=[32, 64, 80], selection_seeds=[654]))
            effective = apply_selection_policy(config, path, root, 'protocol')
            self.assertEqual(effective['seeds'], [654])
            self.assertEqual(effective['selection_seeds'], [654])
            self.assertEqual((root / 'selection_policy.json').read_bytes(), before)
            self.assertEqual(read_json(other), {'preserved': True})
            self.assertEqual(config['seeds'], [654, 114, 25, 759, 281])

    def test_replay_epoch_change_gets_new_directory_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / 'supcon_selected/fold_1/seed_654'
            write_json(old / 'request.json', {'epochs': 7})
            tasks = [dict(output_dir=str(old), identity=dict(fold_id=1))]
            compatible_stage_paths(tasks, root, 654, 'supcon_selected', 9)
            self.assertIn('stage1epoch9', tasks[0]['output_dir'])
            self.assertEqual(read_json(old / 'request.json'), {'epochs': 7})

    def test_explicit_window_expansion_preserves_policies_and_is_blocked_after_fit(self):
        for started in [False, True]:
            with self.subTest(started=started), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = dict(window_lengths=[16, 32, 64, 80, 96], seeds=[654, 114])
                path = root / 'config.json'
                write_json(path.with_suffix('.selection.json'),
                           dict(window_lengths=[32, 64, 80], selection_seeds=[654]))
                apply_selection_policy(config, path, root, 'protocol')
                before = (root / 'selection_policy.json').read_bytes()
                write_json(path.with_suffix('.selection.json'), dict(
                    window_lengths=[16, 32, 64, 80, 96], selection_seeds=[654], allow_window_expansion=True))
                if started:
                    (root / 'window_evaluation').mkdir()
                    with self.assertRaisesRegex(ValueError, 'after final'):
                        apply_selection_policy(config, path, root, 'protocol')
                    self.assertEqual((root / 'selection_policy.json').read_bytes(), before)
                else:
                    effective = apply_selection_policy(config, path, root, 'protocol')
                    self.assertEqual(effective['window_lengths'], [16, 32, 64, 80, 96])
                    self.assertEqual(effective['seeds'], [654])
                    self.assertEqual(len(list((root / 'policy_history').glob('*.json'))), 2)
                    self.assertEqual(apply_selection_policy(config, path, root, 'protocol'), effective)
                    write_json(path.with_suffix('.selection.json'), dict(
                        window_lengths=[16, 32, 64, 80, 96, 128], selection_seeds=[654],
                        allow_window_expansion=True, allow_new_window_lengths=True))
                    expanded = apply_selection_policy(config, path, root, 'protocol')
                    self.assertIn(128, expanded['window_lengths'])
                    self.assertEqual(config['window_lengths'], [16, 32, 64, 80, 96])
                    path.with_suffix('.selection.json').unlink()
                    self.assertEqual(apply_selection_policy(config, path, root, 'protocol'), expanded)

    def test_all_three_windows_select_fit_test_and_resume_without_changing_core_identity(self):
        self.exercise_windows('none')

    def test_all_three_reduced_windows_select_fit_test_and_resume(self):
        self.exercise_windows('mean')

    def test_balanced_all_windows_fit_test_resume(self):
        self.exercise_windows('none', balanced=True)

    def exercise_windows(self, reduction, balanced=False):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp) / ('without_reduction' if reduction == 'none' else 'with_reduction')
            params = small_params()
            config = dict(window_lengths=[8, 16, 24], seeds=[654], selection_seeds=[654],
                          reduction=reduction, stride=4, cv=dict(n_folds=2, seed=42),
                          recipes={k: copy.deepcopy(params) for k in
                                   ['baseline', 'supcon_stage1', 'supcon_stage2']})
            if balanced:
                config['metric_aggregation'] = 'class_first_patient_recall'
                for key in ('baseline', 'supcon_stage2'):
                    config['recipes'][key].update(class_weighting='balanced',
                                                  metric_aggregation=config['metric_aggregation'])
            manifest = dict(patients=[1, 2], hash='manifest')
            write_json(root / 'protocol.json', dict(protocol_hash='protocol', model_hash='model',
                                                    manifest_hash='manifest'))
            rng = np.random.default_rng(0)
            for length in config['window_lengths']:
                raw = dict(protocol_hash='protocol', manifest_hash='manifest', portion='development',
                           length=length, stride=4, aggregation='any',
                           normalization='per_frame_hip_shoulder_then_div100',
                           short_samples='pad_each_contiguous_run', schema=1)
                records = [dict(sample=rng.normal(size=(2, 3, 6)).astype('float32'),
                                mask=np.ones((2, 3), dtype=bool),
                                meta=dict(key=f'{p}_{i}', pid=p, label=i % 2, portion='development'))
                           for p in [1, 2] for i in range(4)]
                write_store(root / f'data/L{length}/development/embeddings', records,
                            embedding_provenance(raw, raw, reduction))
            # Finish a smaller search first, then expand without rerunning its fits.
            subset = dict(config, window_lengths=[8, 16])
            previous = select_single_seed(subset, manifest, root, 'protocol', 'cpu')
            checkpoints = {p: p.stat().st_mtime_ns for p in root.glob('cv/**/checkpoint.pt')}
            plan = select_single_seed(config, manifest, root, 'protocol', 'cpu')
            self.assertEqual(checkpoints, {p: p.stat().st_mtime_ns for p in checkpoints})
            archive = root / 'policy_history' / f"validation_results_{previous['hash']}.json"
            self.assertEqual(read_json(archive), previous)
            for method in ['baseline', 'supcon']:
                self.assertEqual(set(plan['methods'][method]), {'8', '16', '24'})
            self.assertFalse((root / 'selection.json').exists())
            self.assertTrue(all('seed_654' in str(p) for p in root.glob('cv/**/run.json')))
            # Before all fits finish, not even the first test routine may run.
            with patch('scripts.sere_window_evaluation.test') as forbidden:
                with self.assertRaises(FileNotFoundError):
                    evaluate_all(config, manifest, root, 'protocol', 'cpu')
                forbidden.assert_not_called()
            final = fit_all_windows(config, manifest, root, 'protocol', 'cpu')
            self.assertEqual(set(final['windows']), {'8', '16', '24'})
            for length in config['window_lengths']:
                child = root / f'window_evaluation/L{length}'
                result = read_json(child / 'final_models.json')
                for method in ['baseline', 'supcon']:
                    self.assertEqual(set(result['runs'][method]), {'654'})
                records = [dict(sample=rng.normal(size=(2, 3, 6)).astype('float32'),
                                mask=np.ones((2, 3), dtype=bool),
                                meta=dict(key=f'{p}_{i}', pid=p, label=i % 2, portion='test'))
                           for p in [1, 2] for i in range(4)]
                write_store(child / f'data/L{length}/test/embeddings', records,
                            dict(portion='test', reduction=reduction))
            with patch('stamp.sere.workflow.prepare_portions'):
                result = evaluate_all(config, manifest, root, 'protocol', 'cpu')
            self.assertEqual(len(result['rows']), 6)
            if balanced:
                for row in result['rows']:
                    self.assertEqual(row['test_always_positive_ba'], .5)
                    self.assertIn('validation_negative_recall', row)
                    self.assertIn('test_confusion_matrix', row)
                request = read_json(root / 'cv/L8/baseline/fold_1/seed_654/request.json')
                self.assertEqual(request['class_balance']['class_counts'], [2, 2])
                head = read_json(root / 'cv/L8/head_curves/fold_1/seed_654/request.json')
                self.assertIn('class_balance', head)
                stage1 = read_json(root / 'cv/L8/supcon_curves/fold_1/seed_654/request.json')
                self.assertNotIn('class_balance', stage1)
            self.assertEqual({r['window_length'] for r in result['rows']}, {8, 16, 24})
            self.assertTrue(all('validation_fold_std_ba' in r and 'test_ba' in r for r in result['rows']))
            self.assertTrue((root / 'evaluation_results.csv').exists())
            self.assertEqual(plan, select_single_seed(config, manifest, root, 'protocol', 'cpu'))
            self.assertEqual(final, fit_all_windows(config, manifest, root, 'protocol', 'cpu'))
            self.assertEqual(result, evaluate_all(config, manifest, root, 'protocol', 'cpu'))

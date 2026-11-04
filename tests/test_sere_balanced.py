"""Regression checks for single-class scoring and training-only class weights."""
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.sere_single_seed import aggregate
from stamp.sere.metrics import CLASS_FIRST, classification_metrics, patient_class_metrics, aggregate_test
from stamp.sere.training import training_class_balance


class BalancedTests(unittest.TestCase):
    def test_always_positive_no_longer_scores_7778(self):
        patients = {str(p): classification_metrics([1] if p < 10 else [0, 1],
                                                  [1] if p < 10 else [1, 1]) for p in range(18)}
        self.assertAlmostEqual(np.mean([r['ba'] for r in patients.values()]), 14 / 18)
        result = patient_class_metrics(patients)
        self.assertEqual(result['ba'], .5)
        self.assertEqual(result['negative_recall'], 0)
        self.assertEqual(result['positive_recall'], 1)
        self.assertEqual(result['predicted_positive_rate'], 1)
        self.assertEqual(result['pooled_ba'], .5)
        self.assertIsNone(patients['0']['negative_recall'])
        self.assertEqual(aggregate_test({'654': {'patients': patients}}, list(patients),
                                        [654], CLASS_FIRST)['mean_ba'], .5)

    def test_epoch_and_fold_sd_use_corrected_metric(self):
        folds = [dict(fold_id=1, validation_patients=[1, 2, 3]),
                 dict(fold_id=2, validation_patients=[4, 5])]
        # Three positive-only patients, one mixed patient per fold.
        rows = [
            {'1': [[0, 0], [0, 10]], '2': [[0, 0], [0, 10]], '3': [[0, 10], [0, 10]],
             '4': [[0, 0], [0, 10]], '5': [[0, 10], [0, 10]]},
            {'1': [[0, 0], [5, 5]], '2': [[0, 0], [5, 5]], '3': [[10, 0], [5, 5]],
             '4': [[0, 0], [5, 5]], '5': [[5, 5], [5, 5]]}]
        runs = {}
        for fold in folds:
            curve = []
            for row in rows:
                patients = {}
                for pid in fold['validation_patients']:
                    m = np.array(row[str(pid)])
                    recalls = [m[c, c] / m[c].sum() for c in (0, 1) if m[c].sum()]
                    patients[str(pid)] = dict(ba=float(np.mean(recalls)), confusion_matrix=m.tolist())
                curve.append(dict(patients=patients))
            runs[f"{fold['fold_id']}/654"] = dict(curve=curve)
        self.assertEqual(aggregate(runs, folds, [1, 2, 3, 4, 5], [654], 'ba')['epoch'], 1)
        corrected = aggregate(runs, folds, [1, 2, 3, 4, 5], [654], 'ba', CLASS_FIRST)
        self.assertEqual(corrected['epoch'], 2)
        self.assertEqual(corrected['score'], .625)
        self.assertEqual(corrected['fold_scores'], {'1': .75, '2': .5})
        self.assertAlmostEqual(corrected['fold_std_ba'], np.std([.75, .5], ddof=1))

    def test_weights_exclude_validation_and_other_store_keys(self):
        store = SimpleNamespace(by_key={str(i): {'label': int(i != 0)} for i in range(100)})
        view = SimpleNamespace(store=store, keys=['0', '1', '2', '3'])
        result = training_class_balance(view)
        self.assertEqual(result['class_counts'], [1, 3])
        np.testing.assert_allclose(result['class_weights'], [2, 2 / 3])
        store.by_key['99']['label'] = 0
        self.assertEqual(training_class_balance(view), result)
        view.keys = ['1', '2']
        with self.assertRaisesRegex(ValueError, 'both classes'):
            training_class_balance(view)

    def test_undefined_group_recall_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'both classes'):
            patient_class_metrics({'1': classification_metrics([1], [1])})

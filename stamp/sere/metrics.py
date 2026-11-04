"""Patient-equal scoring and common-epoch selection (epochs are one-based)."""
import numpy as np


def classification_metrics(truths, predictions):
    truths, predictions = np.asarray(truths), np.asarray(predictions)
    if not len(truths) or truths.shape != predictions.shape:
        raise ValueError('Nonempty aligned predictions required')
    if not np.isin(truths, [0, 1]).all() or not np.isin(predictions, [0, 1]).all():
        raise ValueError('SERE labels must be 0 or 1')
    support = [int((truths == c).sum()) for c in (0, 1)]
    recall = [float((predictions[truths == c] == c).mean()) for c in (0, 1) if support[c]]
    return dict(ba=float(np.mean(recall)), class_support=support,
                negative_recall=float((predictions[truths == 0] == 0).mean()) if support[0] else None,
                positive_recall=float((predictions[truths == 1] == 1).mean()) if support[1] else None,
                predicted_positive_rate=float((predictions == 1).mean()),
                single_class=0 in support,
                confusion_matrix=[[int(((truths == c) & (predictions == p)).sum())
                                   for p in (0, 1)] for c in (0, 1)])


def average_curves(runs, patients, seeds, metric):
    if set(runs) != {str(p) for p in patients}:
        raise ValueError('Missing or unexpected LOPO patients')
    by_patient, expected_length = {}, None
    for patient in patients:
        per_seed = runs[str(patient)]
        if set(per_seed) != {str(s) for s in seeds}:
            raise ValueError(f'Incomplete seed set for patient {patient}')
        curves = []
        for seed in seeds:
            curve = np.asarray([e[metric] for e in per_seed[str(seed)]['curve']], dtype=float)
            if expected_length is None:
                expected_length = len(curve)
            if not len(curve) or len(curve) != expected_length or not np.isfinite(curve).all():
                raise ValueError('Every run must have the same complete, finite epoch curve')
            curves.append(curve)
        by_patient[str(patient)] = np.mean(curves, axis=0).tolist()
    aggregate = np.mean(list(by_patient.values()), axis=0)
    selected = int(np.argmin(aggregate) if metric == 'loss' else np.argmax(aggregate))
    return dict(epoch=selected + 1, score=float(aggregate[selected]),
                curve=aggregate.tolist(), patient_curves=by_patient)


def aggregate_test(seed_results, patients, seeds, aggregation=None):
    if aggregation == CLASS_FIRST:
        diagnostics = {str(s): patient_class_metrics(seed_results[str(s)]['patients']) for s in seeds}
        return dict(mean_ba=float(np.mean([r['ba'] for r in diagnostics.values()])),
                    seed_std=float(np.std([r['ba'] for r in diagnostics.values()])),
                    aggregation=CLASS_FIRST, diagnostics_per_seed=diagnostics,
                    patients_by_seed={str(s): seed_results[str(s)]['patients'] for s in seeds})
    per_patient = {}
    for patient in patients:
        scores = []
        for seed in seeds:
            scores.append(seed_results[str(seed)]['patients'][str(patient)]['ba'])
        per_patient[str(patient)] = dict(mean_ba=float(np.mean(scores)),
                                        seed_std=float(np.std(scores)), seed_scores=scores)
    per_seed = [float(np.mean([seed_results[str(s)]['patients'][str(p)]['ba']
                              for p in patients])) for s in seeds]
    return dict(mean_ba=float(np.mean([m['mean_ba'] for m in per_patient.values()])),
                seed_std=float(np.std(per_seed)), patient_equal_ba_per_seed=per_seed,
                patients=per_patient)


def average_fold_curves(runs, folds, patients, seeds, metric, aggregation=None):
    """Unpack grouped-fold curves BEFORE averaging; folds have unequal sizes."""
    if set(runs) != {str(f['fold_id']) for f in folds}:
        raise ValueError('Incomplete grouped-fold results')
    per_patient = {}
    for fold in folds:
        seed_runs = runs[str(fold['fold_id'])]
        if set(seed_runs) != {str(s) for s in seeds}:
            raise ValueError('Incomplete seed results')
        for pid in fold['validation_patients']:
            if str(pid) in per_patient:
                raise ValueError('Patient appears in multiple validation folds')
            per_patient[str(pid)] = {}
            for seed in seeds:
                rows = seed_runs[str(seed)]['curve']
                for row in rows:
                    if set(row['patients']) != {str(p) for p in fold['validation_patients']}:
                        raise ValueError('Incomplete patient metrics in validation curve')
                per_patient[str(pid)][str(seed)] = dict(curve=[r['patients'][str(pid)] for r in rows])
    result = average_curves(per_patient, patients, seeds, metric)
    if metric == 'ba' and aggregation == CLASS_FIRST:
        diagnostics = []
        for epoch in range(len(result['curve'])):
            diagnostics.append({str(seed): patient_class_metrics({
                str(pid): per_patient[str(pid)][str(seed)]['curve'][epoch] for pid in patients})
                for seed in seeds})
        curve = [float(np.mean([r['ba'] for r in row.values()])) for row in diagnostics]
        selected = int(np.argmax(curve))
        result.update(epoch=selected + 1, score=curve[selected], curve=curve,
                      aggregation=CLASS_FIRST, diagnostics_per_seed=diagnostics[selected])
    return result


CLASS_FIRST = 'class_first_patient_recall'


def patient_class_metrics(patients):
    """Average patients within each supported class, then give both classes 1/2.

    A missing class has undefined recall, never a perfect score. Both classes
    must exist across the evaluated group. Confusion matrices also support
    rescoring historical curves without loading models or test data.
    """
    matrices = [np.asarray(r['confusion_matrix'], dtype=np.int64) for r in patients.values()]
    if not matrices or any(m.shape != (2, 2) or (m < 0).any() or m.sum() == 0 for m in matrices):
        raise ValueError('Nonempty binary confusion matrices required')
    recalls = []
    for c in (0, 1):
        values = [float(m[c, c] / m[c].sum()) for m in matrices if m[c].sum()]
        if not values:
            raise ValueError('Class-first BA requires both classes across the evaluated group')
        recalls.append(float(np.mean(values)))
    pooled = np.sum(matrices, axis=0)
    return dict(ba=float(np.mean(recalls)), negative_recall=recalls[0], positive_recall=recalls[1],
                pooled_ba=float(np.mean(np.diag(pooled) / pooled.sum(axis=1))),
                confusion_matrix=pooled.tolist(), class_support=pooled.sum(axis=1).tolist(),
                predicted_positive_rate=float(pooled[:, 1].sum() / pooled.sum()),
                always_positive_ba=0.5, aggregation=CLASS_FIRST)

"""One-seed selection that reuses unchanged fits and preserves five-seed results."""
import copy
from pathlib import Path

import numpy as np

from stamp.sere.data import file_digest, read_json, write_json
from stamp.sere.metrics import CLASS_FIRST, average_fold_curves, patient_class_metrics
from stamp.sere.workflow import (folds_for, grouped_runs, load_locked, lock_artifact,
                                 tasks_for)
from stamp.sere.workers import dispatch
from stamp.sere.checkpoint import resolve_run

if __package__:
    from .sere_selection_policy import final_work_started, replace_with_history
else:
    from sere_selection_policy import final_work_started, replace_with_history


def valid_expansion(saved, config, output):
    return (saved.get('selection_seeds') == config['selection_seeds']
            and saved.get('evaluation') == 'all_candidate_windows'
            and set(saved.get('window_lengths', [])) < set(config['window_lengths'])
            and not final_work_started(output))


def aggregate(results, folds, patients, seeds, metric, aggregation=None):
    runs = grouped_runs(results, folds, seeds)
    selected = average_fold_curves(runs, folds, patients, seeds, metric, aggregation)
    epoch = selected['epoch'] - 1
    # Within each fold, give its held-out patients equal weight. Folds themselves
    # are the five observations for SD; the selection mean remains patient-equal.
    scores = {str(f['fold_id']): float(np.mean([
        runs[str(f['fold_id'])][str(s)]['curve'][epoch]['patients'][str(pid)][metric]
        for s in seeds for pid in f['validation_patients']])) for f in folds}
    if metric == 'ba' and aggregation == CLASS_FIRST:
        scores = {str(f['fold_id']): float(np.mean([
            patient_class_metrics(runs[str(f['fold_id'])][str(seed)]['curve'][epoch]['patients'])['ba']
            for seed in seeds])) for f in folds}
    selected.update(fold_scores=scores, fold_std_ddof=1,
                    **{f'fold_mean_{metric}': float(np.mean(list(scores.values()))),
                       f'fold_std_{metric}': float(np.std(list(scores.values()), ddof=1))})
    return selected


def compatible_stage_paths(tasks, directory, seed, stage, epoch=None):
    """Only replay/head fits whose initialization changes get new directories."""
    for task in tasks:
        path = Path(task['output_dir']) / 'request.json'
        compatible = True
        if path.exists():
            request = read_json(path)
            if stage == 'supcon_selected':
                compatible = request['epochs'] == epoch
            elif stage == 'head_curves':
                init = task['initialization']
                expected = dict(signature=init['signature'], epochs=init['epochs'],
                                sha256=file_digest(init['path']))
                compatible = request['initialization'] == expected
        if not compatible:
            fid = task['identity']['fold_id']
            task['output_dir'] = str(directory / f'{stage}_seed{seed}_stage1epoch{epoch}' /
                                     f'fold_{fid}' / f'seed_{seed}')
    return tasks


def completed_tasks(tasks):
    """Read compatible saved results for status; no model execution or test data."""
    results = {}
    for task in tasks:
        root = Path(task['output_dir'])
        if not (root / 'run.json').exists():
            return None
        result = read_json(root / 'run.json')
        if result['identity'] != task['identity']:
            raise ValueError(f'Completed result identity differs: {root}')
        if file_digest(root / 'checkpoint.pt') != result['checkpoint_sha256']:
            raise ValueError(f'Completed checkpoint checksum differs: {root}')
        results[task['tag']] = resolve_run(result, root)
    return results


def require_single_seed_selection(config, output, protocol_hash):
    result = load_locked(Path(output) / 'validation_results.json', protocol_hash)
    if result.get('selection_seeds') != config['selection_seeds']:
        raise ValueError('Run select first: completed selection does not use the requested single seed')
    return result


def select_single_seed(config, manifest, output, protocol_hash, devices, report_only=False):
    config = copy.deepcopy(config)
    config['seeds'] = config['selection_seeds']
    seed = config['seeds'][0]
    output = Path(output)
    destination = output / 'validation_results.json'
    if destination.exists() and not report_only:
        saved = load_locked(destination, protocol_hash)
        if (saved.get('selection_seeds') == config['seeds']
                and saved.get('window_lengths') == config['window_lengths']
                and saved.get('evaluation') == 'all_candidate_windows'):
            return saved
        if not valid_expansion(saved, config, output):
            raise ValueError('Completed validation results use a different evaluation policy')
    folds = folds_for(config, manifest)
    candidates = {'baseline': {}, 'supcon': {}}
    progress = {}
    for length in sorted(config['window_lengths']):
        directory = output / 'cv' / f'L{length}'
        summary = dict(protocol_hash=protocol_hash, selection_seeds=config['seeds'],
                       length=length, methods={}, status={})
        def execute(method, stage, params, name, **kwargs):
            tasks = tasks_for(config, manifest, output, protocol_hash, method, length,
                              stage, params, directory / name, **kwargs)
            stage1_epoch = kwargs.get('epochs')
            if name == 'head_curves':
                stage1_epoch = next(iter(kwargs['pretrained'].values()))['epochs']
            tasks = compatible_stage_paths(tasks, directory, seed, name, stage1_epoch)
            summary['status'][name] = dict(completed=sum(
                (Path(t['output_dir']) / 'run.json').exists() for t in tasks), total=len(tasks))
            result = completed_tasks(tasks) if report_only else dispatch(tasks, devices)
            summary['status'][name]['completed'] = sum(
                (Path(t['output_dir']) / 'run.json').exists() for t in tasks)
            return result
        def save():
            write_json(directory / f'validation_summary_seed{seed}.json', summary)
            progress[str(length)] = summary
        try:
            baseline = execute('baseline', 'baseline', config['recipes']['baseline'], 'baseline')
            if baseline is not None:
                chosen = aggregate(baseline, folds, manifest['patients'], config['seeds'], 'ba', config.get('metric_aggregation'))
                candidates['baseline'][str(length)] = chosen
                summary['methods']['baseline'] = chosen
            stage1 = execute('supcon', 'supcon', config['recipes']['supcon_stage1'], 'supcon_curves')
            if stage1 is None:
                continue
            s1 = aggregate(stage1, folds, manifest['patients'], config['seeds'], 'loss')
            summary['stage1'] = s1
            pretrained = execute('supcon', 'supcon', config['recipes']['supcon_stage1'], 'supcon_selected',
                                 validate=False, epochs=s1['epoch'])
            if pretrained is None:
                continue
            head = execute('supcon', 'head', config['recipes']['supcon_stage2'], 'head_curves', pretrained=pretrained)
            if head is not None:
                chosen = aggregate(head, folds, manifest['patients'], config['seeds'], 'ba', config.get('metric_aggregation'))
                chosen['stage1'] = s1
                candidates['supcon'][str(length)] = chosen
                summary['methods']['supcon'] = chosen
        except FileNotFoundError:
            if not report_only:
                raise
            summary['status']['data'] = 'development embeddings not prepared'
        finally:
            save()
    report = dict(protocol_hash=protocol_hash, selection_seeds=config['seeds'], windows=progress)
    write_json(output / f'selection_progress_seed{seed}.json', report)
    if report_only:
        return report
    methods = {}
    for method, results in candidates.items():
        if set(results) != {str(n) for n in config['window_lengths']}:
            raise ValueError('Incomplete single-seed candidate set')
        methods[method] = {}
        for length, chosen in results.items():
            row = dict(length=int(length), epochs=chosen['epoch'], score=chosen['score'],
                       fold_mean_ba=chosen['fold_mean_ba'], fold_std_ba=chosen['fold_std_ba'],
                       fold_scores=chosen['fold_scores'], fold_std_ddof=1)
            if 'diagnostics_per_seed' in chosen:
                row['diagnostics_per_seed'] = chosen['diagnostics_per_seed']
            if method == 'supcon':
                row['stage1_epochs'] = chosen['stage1']['epoch']
            methods[method][length] = row
    payload = dict(
        protocol_hash=protocol_hash, selection_seeds=config['seeds'],
        window_lengths=config['window_lengths'], evaluation='all_candidate_windows',
        reduction=config.get('reduction', 'none'), methods=methods, candidates=candidates,
        folds=folds, aggregation=config.get('metric_aggregation', 'equal_patients_one_fixed_seed_at_each_epoch'),
        variability='sample_std_across_fold_BA_ddof1',
        tie_break='earliest_epoch', test_accessed=False)
    if destination.exists():
        saved = load_locked(destination, protocol_hash)
        if not valid_expansion(saved, config, output):
            raise ValueError('Cannot replace completed validation results')
        return replace_with_history(destination, saved, payload)
    return lock_artifact(destination, payload)

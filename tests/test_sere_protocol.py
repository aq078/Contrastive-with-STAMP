import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from stamp.sere.data import (create_manifest, fold_keys, iter_windows, source_paths,
                             store_index, write_store, SEEDS)
from stamp.sere.embedding import (EmbeddingStore, masked_moment_embeddings,
                                  arrange_embeddings, embed_store)
from stamp.sere.metrics import average_curves, classification_metrics
from stamp.sere.training import build_model, initialize_from_supcon, run_stage, training_mode
from stamp.sere.workflow import (identity, initialization, initialize, prepare_portions,
                                 reduction_output, select, fit, test as evaluate_test)


def small_params():
    return dict(D=8, encoder_aggregation='attention_pooling', use_batch_norm=False,
                use_instance_norm=True, initial_proj_params={'type': 'full', 'dropout_rate': 0.2},
                final_classifier_params=None,
                pe_params=dict(pe_type='basic', use_token_positional_embeddings=True,
                               use_spatial_positional_embeddings=True, use_temporal_positional_embeddings=True),
                transformer_params=None,
                gated_mlp_params=dict(type='criss_cross', n_layers=2, dim_feedforward=16,
                                     dropout_rate=0.2, combination_mode='concat', recurrent=False),
                mhap_params=dict(A=2, dropout_rate=0.2, n_queries_per_head=2,
                                 query_combination='weighted_sum', lambda_for_residual=0.1),
                n_epochs=3, train_batch_size=2, test_batch_size=2,
                lr_params=dict(use_scheduler=True, scheduler_type='one_cycle', initial_lr=0.001, max_lr=0.002),
                optimizer_params=dict(optimizer_name='adamw', weight_decay=0.01),
                label_smoothing=0.1, use_gradient_clipping=True,
                supcon_params=dict(temperature=0.07, proj_head=dict(proj_dim=4, proj_hidden_dim=8)),
                aug_params=dict(jitter_std=0.01))


class TemporalDataTests(unittest.TestCase):
    def make_source(self, root):
        for ex in ['E1', 'E3']:
            paths = source_paths(root, ex)
            for path in paths.values():
                path.parent.mkdir(parents=True, exist_ok=True)
            rows, labels, trials = [], [], []
            for pid in [1, 2]:
                affected = 0 if ex == 'E1' else -1
                # Shared endpoints must not become development/test duplicates.
                for trial, lo, hi in [(1, 0, 19), (2, 19, 38)]:
                    row = dict(pid=pid, trial=trial, frame_init=lo, frame_end=hi)
                    if ex == 'E1':
                        row['affected'] = affected
                    trials.append(row)
                for frame in range(39):
                    row = dict(pid=pid, affected=affected, frame=frame)
                    for j in range(33):
                        for axis in 'xyz':
                            row[f'{j}{axis}'] = j + frame * 0.01
                    rows.append(row)
                    labels.append(dict(pid=pid, affected=affected, frame=frame, comp=int(frame % 2 == 0)))
            pd.DataFrame(rows).to_csv(paths['world'], index=False)
            pd.DataFrame(labels).to_csv(paths['frames'], index=False)
            pd.DataFrame(trials).to_csv(paths['trials'], index=False)

    def test_split_before_windowing_padding_and_cross_trial_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_source(tmp)
            manifest = create_manifest(tmp, ['E1', 'E3'])
            self.assertEqual(manifest['patients'], [1, 2])
            for length in [3, 8, 32]:
                sets = {}
                for portion in ['development', 'test']:
                    windows = list(iter_windows(tmp, manifest, portion, length, 2))
                    self.assertEqual({r['meta']['exercise'] for r in windows}, {'E1', 'E3'})
                    frame_ids = set()
                    for r in windows:
                        m = r['meta']
                        self.assertEqual(r['sample'].shape, (99, length))
                        self.assertEqual(int(r['mask'].sum()), m['valid_length'])
                        self.assertTrue(np.all(r['sample'][:, ~r['mask']] == 0))
                        for f in range(m['frame_start'], m['frame_stop']):
                            self.assertNotEqual(f, 19)
                            frame_ids.add((m['exercise'], m['pid'], m['affected'], f))
                    sets[portion] = frame_ids
                self.assertFalse(sets['development'] & sets['test'])

    def test_missing_frames_break_runs_and_cuts_do_not_move(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_source(tmp)
            path = source_paths(tmp, 'E1')['world']
            frame = pd.read_csv(path)
            frame.loc[frame.frame == 7, '0x'] = np.nan
            frame.to_csv(path, index=False)
            manifest = create_manifest(tmp, ['E1'])
            self.assertEqual(manifest['trials'][0]['cut'], 16)
            for r in iter_windows(tmp, manifest, 'development', 32, 2):
                m = r['meta']
                self.assertFalse(m['frame_start'] <= 7 < m['frame_stop'])

    def test_identical_label_duplicates_are_safe_but_conflicts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_source(tmp)
            path = source_paths(tmp, 'E1')['frames']
            df = pd.read_csv(path)
            pd.concat([df, df]).to_csv(path, index=False)
            manifest = create_manifest(tmp, ['E1'])
            self.assertTrue(list(iter_windows(tmp, manifest, 'development', 8, 2)))
            conflicting = df.iloc[[0]].copy()
            conflicting['comp'] = 1 - conflicting['comp']
            pd.concat([df, conflicting]).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                list(iter_windows(tmp, manifest, 'development', 8, 2))

    def test_fold_isolation_and_test_rejection(self):
        index = [dict(key=f'k{p}', pid=p, portion='development') for p in [1, 2, 3]]
        self.assertEqual(fold_keys(index, 2, 'val'), ['k2'])
        self.assertEqual(fold_keys(index, 2, 'train'), ['k1', 'k3'])
        self.assertEqual(fold_keys(index, role='final'), ['k1', 'k2', 'k3'])
        with self.assertRaises(ValueError):
            fold_keys([dict(key='test', pid=1, portion='test')], 1, 'val')
        with self.assertRaises(FileNotFoundError):
            with tempfile.TemporaryDirectory() as tmp:
                prepare_portions({}, {}, Path(tmp), 'hash', [8], 'test', 'cpu')

    def test_reduction_tracks_share_splits_but_not_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.make_source(root / 'source')
            (root / 'moment').mkdir()
            (root / 'moment' / 'config.json').write_text('{"patch_len": 8}')
            config = dict(protocol='sere-temporal80-lopo-v1', reduction='none', seeds=SEEDS,
                          window_lengths=[8, 16], stride=4, exercises=['E1', 'E3'], cv={'n_folds': 2, 'seed': 42},
                          source_root=str(root / 'source'), moment_model_dir=str(root / 'moment'),
                          output_dir=str(root / 'runs'),
                          recipes={k: small_params() for k in ['baseline', 'supcon_stage1', 'supcon_stage2']})
            path = root / 'config.json'
            path.write_text(json.dumps(config))
            a, manifest_a, output_a, hash_a = initialize(path, reduction='none')
            b, manifest_b, output_b, hash_b = initialize(path, reduction='mean')
            self.assertEqual(manifest_a, manifest_b)
            self.assertNotEqual(hash_a, hash_b)
            self.assertEqual(output_a.name, 'without_reduction')
            self.assertEqual(output_b.name, 'with_reduction')
            self.assertEqual(initialize(path, reduction='none')[3], hash_a)
            self.assertEqual(json.loads(path.read_text()), config)
            with self.assertRaises(ValueError):
                reduction_output(dict(config, reduction='unsupported'))


class MetricsTests(unittest.TestCase):
    def test_single_class_and_patient_equal_common_epoch(self):
        self.assertAlmostEqual(classification_metrics([1] * 10, [1] * 8 + [0] * 2)['ba'], 0.8)
        runs = {
            '1': {'5': {'curve': [{'ba': 1.0}, {'ba': 0.5}]},
                  '6': {'curve': [{'ba': 1.0}, {'ba': 0.5}]}},
            '2': {'5': {'curve': [{'ba': 0.0}, {'ba': 0.8}]},
                  '6': {'curve': [{'ba': 0.2}, {'ba': 0.8}]}}}
        result = average_curves(runs, [1, 2], [5, 6], 'ba')
        self.assertEqual(result['epoch'], 2)
        self.assertAlmostEqual(result['score'], 0.65)
        del runs['2']['6']
        with self.assertRaises(ValueError):
            average_curves(runs, [1, 2], [5, 6], 'ba')


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_padding_cannot_change_valid_predictions_or_gradients(self):
        model = build_model(small_params(), (4, 3, 6), 'baseline').eval()
        x = torch.randn(2, 4, 3, 6, requires_grad=True)
        mask = torch.ones(2, 4, 3, dtype=torch.bool)
        mask[:, 2:] = False
        y = model(x, valid_mask=mask)[0]
        altered = x.detach().clone()
        altered[:, 2:] = 1e6
        torch.testing.assert_close(y, model(altered, valid_mask=mask)[0])
        y.sum().backward()
        self.assertEqual(float(x.grad[:, 2:].abs().sum()), 0)
        # Fully valid masks preserve the legacy model behavior.
        torch.testing.assert_close(model(x.detach())[0],
                                   model(x.detach(), valid_mask=torch.ones_like(mask))[0], atol=1e-6, rtol=1e-5)

    def test_reduction_preserves_coordinate_joint_order_and_excludes_padding(self):
        channels = torch.arange(99, dtype=torch.float32)
        features = channels[None, None, :, None].expand(2, 3, 99, 2).clone()
        features[:, 1] += 1000
        features[:, 2] = 1e6
        valid = torch.ones(2, 3, 99, dtype=torch.bool)
        valid[:, 2] = False
        valid[1, 1] = False
        plain, plain_mask = arrange_embeddings(features, valid, 'none')
        self.assertIs(plain, features)
        self.assertIs(plain_mask, valid)
        reduced, mask = arrange_embeddings(features, valid, 'mean')
        self.assertEqual(reduced.shape, (2, 3, 33, 2))
        self.assertTrue(mask.all())
        for coordinate in range(3):
            for joint in range(33):
                self.assertEqual(reduced[0, coordinate, joint, 0], joint * 3 + coordinate + 500)
                self.assertEqual(reduced[1, coordinate, joint, 0], joint * 3 + coordinate)
        with self.assertRaises(ValueError):
            arrange_embeddings(features, torch.zeros_like(valid), 'mean')

    def test_embedding_cache_records_and_checks_reduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = dict(key='sample', pid=1, label=0, portion='development')
            provenance = {'portion': 'development', 'length': 9}
            write_store(root / 'raw', [dict(sample=np.ones((99, 9), dtype='float32'),
                                           mask=np.ones(9, dtype=bool), meta=meta)], provenance)
            features = torch.arange(99, dtype=torch.float32).reshape(1, 1, 99, 1).expand(1, 2, 99, 6)
            valid = torch.ones(1, 2, 99, dtype=torch.bool)
            model = SimpleNamespace(patch_len=8)
            with patch('stamp.sere.embedding.masked_moment_embeddings', return_value=(features, valid)):
                embed_store(root / 'raw', root / 'mean', model, 'cpu', provenance, reduction='mean')
            index, stored = store_index(root / 'mean')
            self.assertEqual(stored['reduction'], 'mean')
            self.assertEqual(stored['layout'], '3x33')
            store = EmbeddingStore(root / 'mean')
            try:
                self.assertEqual(store.shape, (3, 33, 6))
                self.assertEqual(store.read('sample')['meta'], meta)
            finally:
                store.close()
            with self.assertRaises(ValueError):
                embed_store(root / 'raw', root / 'mean', model, 'cpu', provenance, reduction='none')

    def test_stage2_only_mhap_and_classifier_update(self):
        model = build_model(small_params(), (4, 3, 6), 'head')
        training_mode(model, 'head')
        self.assertFalse(model.gated_mlp.training)
        self.assertFalse(model.linear.training)
        self.assertTrue(model.multi_head_attention_pooling.training)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        x, mask = torch.randn(2, 4, 3, 6), torch.ones(2, 4, 3, dtype=torch.bool)
        torch.nn.functional.cross_entropy(model(x, valid_mask=mask)[0], torch.tensor([0, 1])).backward()
        opt.step()
        changed = []
        for k, value in model.state_dict().items():
            if not torch.equal(value, before[k]):
                changed.append(k)
                self.assertTrue(k.startswith(('classifier.', 'multi_head_attention_pooling.')), k)
        self.assertTrue(any(k.startswith('classifier.') for k in changed))
        self.assertTrue(any(k.startswith('multi_head_attention_pooling.') for k in changed))

    def test_partial_moment_patch_keeps_real_frames_only(self):
        from momentfm.models.layers.revin import RevIN
        from momentfm.models.layers.embed import PatchEmbedding, Patching
        class Encoder(torch.nn.Module):
            def forward(self, inputs_embeds, attention_mask):
                self.mask = attention_mask
                return SimpleNamespace(last_hidden_state=inputs_embeds)
        model = SimpleNamespace(patch_len=8, tokenizer=Patching(8, 8),
                                normalizer=RevIN(1), patch_embedding=PatchEmbedding(
                                    d_model=6, patch_len=8, stride=8, patch_dropout=0), encoder=Encoder())
        x = torch.randn(2, 3, 16)
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[0, :1] = True
        mask[1, :9] = True
        a, valid = masked_moment_embeddings(model, x, mask)
        x.masked_fill_(~mask[:, None, :], 1e6)
        b, _ = masked_moment_embeddings(model, x, mask)
        torch.testing.assert_close(a, b)
        self.assertEqual(valid[:, :, 0].tolist(), [[True, False], [True, True]])
        self.assertTrue(torch.isfinite(a).all())

    def make_store(self, tmp):
        rng = np.random.default_rng(42)
        rows = []
        for pid in [1, 2]:
            for i in range(4):
                rows.append(dict(sample=rng.normal(size=(4, 3, 6)).astype('float32'),
                                 mask=np.ones((4, 3), dtype=bool),
                                 meta=dict(key=f'{pid}_{i}', pid=pid, label=i % 2, portion='development')))
        path = Path(tmp) / 'data'
        write_store(path, rows, {'portion': 'development'})
        return EmbeddingStore(path)

    def test_full_curve_replay_final_fit_and_checkpoint_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self.make_store(tmp)
            try:
                params = small_params()
                ident = identity('protocol', 'supcon', 16, 1, 654)
                full = run_stage(params, 'supcon', store.view(1, 'train'), store.view(1, 'val'),
                                 ident, Path(tmp) / 'full', 'cpu')
                replay = run_stage(params, 'supcon', store.view(1, 'train'), None,
                                   ident, Path(tmp) / 'replay', 'cpu')
                a = torch.load(full['checkpoint'], weights_only=True)['model_state_dict']
                b = torch.load(replay['checkpoint'], weights_only=True)['model_state_dict']
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
                head = run_stage(params, 'head', store.view(1, 'train'), store.view(1, 'val'), ident,
                                 Path(tmp) / 'head', 'cpu', initialization=initialization(replay))
                self.assertEqual(len(head['curve']), 3)
                model = build_model(params, store.shape, 'head')
                with self.assertRaises(ValueError):
                    initialize_from_supcon(model, initialization(replay), dict(ident, held_out_patients=[2]), 'cpu')
                with self.assertRaises(ValueError):
                    initialize_from_supcon(model, initialization(replay), dict(ident, reduction='mean'), 'cpu')
                final = run_stage(params, 'baseline', store.view(role='final'), None,
                                  identity('protocol', 'baseline', 16, None, 654, 'final'),
                                  Path(tmp) / 'final', 'cpu', epochs=2)
                self.assertEqual(final['epochs'], 2)
                self.assertNotIn('ba', final['curve'][0])
                state = torch.load(final['checkpoint'], weights_only=True)
                self.assertEqual(state['horizon'], 3)
                # Opening the database for batches never resets a filtered view.
                self.assertEqual(store.view(1, 'val').keys, ('1_0', '1_1', '1_2', '1_3'))
            finally:
                store.close()

    def test_both_methods_complete_selection_fit_test_and_resume(self):
        self.exercise_workflow('none')

    def test_reduced_both_methods_complete_selection_fit_test_and_resume(self):
        self.exercise_workflow('mean')

    def exercise_workflow(self, reduction):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            params = small_params()
            config = dict(window_lengths=[8, 16], seeds=[654, 114, 25, 759, 281], reduction=reduction, cv={'n_folds': 2, 'seed': 42},
                          recipes={k: copy.deepcopy(params) for k in
                                   ['baseline', 'supcon_stage1', 'supcon_stage2']})
            manifest = {'patients': [1, 2]}
            (root / 'protocol.json').write_text(json.dumps({'protocol_hash': 'protocol'}))
            rng = np.random.default_rng(15)
            shape = (3, 33, 6) if reduction == 'mean' else (2, 99, 6)
            for length in config['window_lengths']:
                records = []
                for pid in manifest['patients']:
                    for i in range(4):
                        records.append(dict(sample=rng.normal(size=shape).astype('float32'),
                                            mask=np.ones(shape[:2], dtype=bool),
                                            meta=dict(key=f'{pid}_{i}', pid=pid, label=i % 2,
                                                      portion='development')))
                write_store(root / 'data' / f'L{length}' / 'development' / 'embeddings',
                            records, {'portion': 'development', 'reduction': reduction, 'protocol_hash': 'protocol'})
            chosen = select(config, manifest, root, 'protocol', 'cpu')
            self.assertFalse((root / 'test').exists())
            self.assertEqual(set(chosen['methods']), {'baseline', 'supcon'})
            self.assertEqual(chosen['reduction'], reduction)
            final = fit(config, manifest, root, 'protocol', 'cpu')
            self.assertEqual(len(final['runs']['baseline']), 5)
            self.assertEqual(len(final['runs']['supcon']), 5)
            # Synthetic test embeddings are introduced only after all fits.
            for length in {m['length'] for m in chosen['methods'].values()}:
                records = []
                for pid, count in [(1, 2), (2, 6)]:
                    for i in range(count):
                        records.append(dict(sample=rng.normal(size=shape).astype('float32'),
                                            mask=np.ones(shape[:2], dtype=bool),
                                            meta=dict(key=f'{pid}_{i}', pid=pid, label=i % 2, portion='test')))
                write_store(root / 'data' / f'L{length}' / 'test' / 'embeddings', records,
                            {'portion': 'test', 'reduction': reduction})
            with patch('stamp.sere.workflow.prepare_portions'):
                results = evaluate_test(config, manifest, root, 'protocol', 'cpu')
            self.assertEqual(set(results['methods']), {'baseline', 'supcon'})
            self.assertEqual(results['reduction'], reduction)
            for method in results['methods'].values():
                self.assertEqual(len(method['patient_equal_ba_per_seed']), 5)
                self.assertAlmostEqual(method['mean_ba'], np.mean(method['patient_equal_ba_per_seed']))
            # Completed artifacts return unchanged; no retraining or test reevaluation.
            self.assertEqual(chosen, select(config, manifest, root, 'protocol', 'cpu'))
            self.assertEqual(final, fit(config, manifest, root, 'protocol', 'cpu'))
            self.assertEqual(results, evaluate_test(config, manifest, root, 'protocol', 'cpu'))


if __name__ == '__main__':
    unittest.main()

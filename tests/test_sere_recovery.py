import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.recover_sere_embeddings import complete_store, recover, recovery_plan, verify_membership
from stamp.sere.data import (SEEDS, PROTOCOL, digest, fold_keys, iter_windows, read_json,
                            store_index, write_json, write_store)
from stamp.sere.embedding import embedding_provenance
from stamp.sere.workflow import initialize, development_path
import test_sere_protocol as fixtures


class RecoveryTests(unittest.TestCase):
    def test_rebuild_preserves_legacy_provenance_and_rejects_changed_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures.TemporalDataTests().make_source(root / 'source')
            (root / 'moment').mkdir()
            (root / 'moment/config.json').write_text('{}')
            config = dict(protocol=PROTOCOL, seeds=SEEDS, reduction='none', window_lengths=[8],
                          stride=4, exercises=['E1', 'E3'], cv=dict(n_folds=2, seed=42),
                          source_root=str(root / 'source'), moment_model_dir=str(root / 'moment'),
                          output_dir=str(root / 'runs'), prepared_data_dir=str(root / 'recovered'),
                          recipes={k: fixtures.small_params() for k in
                                   ['baseline', 'supcon_stage1', 'supcon_stage2']})
            config_path = root / 'config.json'
            write_json(config_path, config)
            config, manifest, output, protocol_hash = initialize(config_path)
            raw = dict(protocol_hash='historical', manifest_hash=manifest['hash'], portion='development',
                       length=8, stride=4, aggregation='any',
                       normalization='per_frame_hip_shoulder_then_div100',
                       short_samples='pad_each_contiguous_run', schema=1)
            records = list(iter_windows(config['source_root'], manifest, 'development', 8, 4))
            index = [r['meta'] for r in records]
            held = [manifest['patients'][0]]
            request = dict(identity=dict(protocol_hash=protocol_hash, reduction='none', length=8,
                                         held_out_patients=held), data=embedding_provenance(raw, raw, 'none'),
                           train_keys_hash=digest(fold_keys(index, held, 'train')),
                           val_keys_hash=digest(fold_keys(index, held, 'val')))
            path = output / 'cv/L8/baseline/fold_1/seed_654/request.json'
            write_json(path, request)
            write_json(path.with_name('run.json'), dict(signature=digest(request)))
            before = path.read_bytes()
            recover(config_path, 'none', 'cpu', raw_only=True)
            destination = root / 'recovered/without_reduction/data/L8/development'
            self.assertEqual(store_index(destination / 'raw'), (index, raw))
            self.assertFalse((destination / 'embeddings').exists())
            # Exercise the complete recovery path without requiring MOMENT/GPU.
            def fake_embed(raw_path, target, model, device, provenance, **kwargs):
                return write_store(target, records, embedding_provenance(provenance, provenance, 'none'))
            with patch('scripts.recover_sere_embeddings.load_moment', return_value=object()), \
                 patch('scripts.recover_sere_embeddings.embed_store', side_effect=fake_embed):
                recover(config_path, 'none', 'cpu')
            self.assertEqual(development_path(config, manifest, output, 8), destination / 'embeddings')
            with patch('scripts.recover_sere_embeddings.load_moment', side_effect=AssertionError('must skip')):
                recover(config_path, 'none', 'cpu')
            self.assertEqual(path.read_bytes(), before)
            replay = dict(request, val_keys_hash=None)
            verify_membership(index, [(path, replay)], 8, manifest['patients'])
            with self.assertRaisesRegex(ValueError, 'sample keys differ'):
                verify_membership(list(reversed(index)), [(path, request)], 8, manifest['patients'])
            request['data']['stride'] = 99
            write_json(path, request)
            with self.assertRaisesRegex(ValueError, 'preprocessing differs'):
                recovery_plan(config, manifest, output, protocol_hash)

    def test_incomplete_recovery_staging_is_rebuilt_but_completed_target_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / 'embeddings'
            staging = target.with_name('embeddings.rebuilding')
            record = dict(sample=np.zeros((1, 2), dtype=np.float32), mask=np.ones(1, dtype=bool),
                          meta=dict(key='one', pid=1, portion='development'))
            def interrupted():
                yield record
                raise RuntimeError('interrupted')
            with self.assertRaises(RuntimeError):
                write_store(staging, interrupted(), {'recipe': 1})
            complete_store(target, {'recipe': 1}, lambda p: write_store(p, [record], {'recipe': 1}))
            self.assertFalse(staging.exists())
            self.assertEqual(store_index(target)[0], [record['meta']])
            with self.assertRaisesRegex(ValueError, 'provenance differs'):
                complete_store(target, {'recipe': 2}, lambda p: self.fail('must not overwrite'))


if __name__ == '__main__':
    unittest.main()

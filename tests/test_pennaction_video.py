import contextlib
import copy
import io
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import lmdb
import numpy as np
from scipy.io import savemat
import torch

from stamp.pennaction.data import build_manifest, normalized_video, supplied_split
from stamp.pennaction.embedding import reduce_video, VideoDataset, prepare
from stamp.pennaction.training import metrics, build_model, fit_task
from stamp.pennaction.workflow import train, test as evaluate_test, summarize
from stamp.sere.checkpoint import atomic_torch_save
from stamp.sere.data import file_digest, read_json, write_json
from test_sere_protocol import small_params


class PennActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_reduction_preserves_joint_coordinate_order_and_ignores_padding(self):
        x = torch.arange(26.).reshape(1, 26, 1).repeat(3, 1, 4)
        x[2] = 10000
        mask = torch.ones(3, 26, dtype=torch.bool)
        mask[2] = False
        reduced, valid = reduce_video(x, mask, 'mean')
        self.assertEqual(tuple(reduced.shape), (2, 13, 4))
        self.assertEqual(reduced[1, 7, 0].item(), 15)
        self.assertTrue(valid.all())
        untouched, _ = reduce_video(x, mask, 'none')
        self.assertEqual(tuple(untouched.shape), (3, 26, 4))

    def test_full_video_normalization_retains_all_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'video.mat'
            n = 19
            x = np.tile(np.arange(13), (n, 1)).astype(float)
            y = x.copy()
            x[:, 0] = np.arange(n)
            visible = np.ones((n, 13))
            visible[4:6, 0] = 0
            bbox = np.tile([0, 0, 20, 20], (n-1, 1))  # missing last bbox is repaired
            savemat(path, dict(x=x, y=y, visibility=visible, bbox=bbox, nframes=n))
            result = normalized_video(path)
            self.assertEqual(result.shape, (26, n))
            self.assertTrue(np.isfinite(result).all())
            self.assertAlmostEqual(float(result[0, 4]), -.3, places=6)

    def test_prepare_is_one_video_resumable_and_does_not_open_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / 'source/labels'
            labels.mkdir(parents=True)
            videos = []
            for portion in ('train', 'val', 'test'):
                path = labels / f'{portion}.mat'
                path.write_bytes(b'fixture')
                videos.append(dict(video=portion, split=portion, label=0, sha256=file_digest(path)))
            config = dict(source_root=str(root/'source'), moment_model_dir='unused', patch_len=8)
            manifest = dict(videos=videos)
            def embed(model, samples, mask):
                self.assertEqual(tuple(samples.shape), (1,26,24))
                self.assertEqual(mask.sum().item(), 19)
                return torch.ones(1,3,26,6), torch.ones(1,3,26,dtype=torch.bool)
            with patch('stamp.pennaction.embedding.normalized_video', return_value=np.zeros((26,19),dtype=np.float32)) as normalize, \
                 patch('stamp.pennaction.embedding.load_moment', return_value=SimpleNamespace(patch_len=8)) as moment, \
                 patch('stamp.pennaction.embedding.masked_moment_embeddings', side_effect=embed):
                prepare(config, manifest, root, 'protocol', 'none', ['train','val'], 'cpu')
                self.assertEqual(len(list((root/'embeddings').glob('*.pt'))), 2)
                self.assertEqual({c.args[0].stem for c in normalize.call_args_list}, {'train','val'})
                self.assertEqual(moment.call_count, 1)
                hashes = {p: file_digest(p) for p in (root/'embeddings').glob('*.pt')}
                moment.reset_mock()
                normalize.reset_mock()
                prepare(config, manifest, root, 'protocol', 'none', ['train','val'], 'cpu')
                moment.assert_not_called()
                normalize.assert_not_called()
                self.assertEqual(hashes, {p: file_digest(p) for p in hashes})

    def test_supplied_split_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = lmdb.open(tmp, map_size=1000000)
            with env.begin(write=True) as txn:
                values = {'__sequence_split__': {'0001': 'train', '0002': 'val', '0003': 'test'},
                          '__class_to_idx__': {'a': 0}, '__dataset_meta__': {},
                          '__keys__': {'train': ['0001_w0'], 'val': ['0001_w1', '0002_w0'], 'test': ['0003_w0']}}
                for key, value in values.items():
                    txn.put(key.encode(), pickle.dumps(value))
            env.close()
            with self.assertRaisesRegex(ValueError, 'multiple splits'):
                supplied_split(tmp)

    def test_multiclass_score_and_seed_std(self):
        result = metrics([0, 1, 2, 2], [2, 2, 2, 2], 3)
        self.assertAlmostEqual(result['ba'], 1/3)
        self.assertEqual(result['accuracy'], .5)
        rows = [{k: v for k in ('validation_ba', 'validation_accuracy', 'test_ba', 'test_accuracy')}
                for v in [0., 1., 0., 1., .5]]
        self.assertAlmostEqual(summarize(rows)['test_ba_std'], .5)

    def setup_runs(self, root, mode):
        recipes = {k: copy.deepcopy(small_params()) for k in ('baseline', 'supcon_stage1', 'supcon_stage2')}
        config = dict(seeds=[654,114,25,759,281], padded_frames=24, patch_len=8, recipes=recipes)
        manifest = dict(classes={'a': 0, 'b': 1}, videos=[])
        rng = torch.Generator().manual_seed(3)
        for portion in ('train', 'val', 'test'):
            for index in range(4):
                video = f'{portion}_{index}'
                row = dict(video=video, split=portion, label=index%2, sha256='source')
                manifest['videos'].append(row)
                shape = (2, 26, 6) if mode == 'none' else (2, 13, 6)
                identity = dict(protocol_hash='protocol', reduction=mode, video=video,
                                source_sha256='source', split=portion)
                path = root / 'embeddings' / f'{video}.pt'
                path.parent.mkdir(parents=True, exist_ok=True)
                atomic_torch_save(dict(identity=identity, label=index%2,
                    features=torch.randn(shape, generator=rng), mask=torch.ones(shape[:2], dtype=torch.bool)), path)
                write_json(path.with_suffix('.json'), dict(identity=identity, sha256=file_digest(path)))
        return config, manifest

    def test_end_to_end_both_modes_five_seeds_and_resume(self):
        for mode in ('none', 'mean'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                config, manifest = self.setup_runs(root, mode)
                trained = train(config, manifest, root, 'protocol', mode, ['cpu'])
                self.assertEqual(set(trained['methods']['baseline']), {'654','114','25','759','281'})
                checksums = {p: file_digest(p) for p in root.glob('runs/**/best.pt')}
                self.assertEqual(trained, train(config, manifest, root, 'protocol', mode, ['cpu']))
                self.assertEqual(checksums, {p: file_digest(p) for p in checksums})
                with patch('stamp.pennaction.workflow.prepare') as prepare:
                    result = evaluate_test(config, manifest, root, 'protocol', mode, 'cpu')
                    self.assertEqual(prepare.call_args.args[5], ['test'])
                self.assertEqual(len(result['per_seed']), 10)
                self.assertEqual(len(result['rows']), 2)
                test = read_json(root / 'test/baseline/seed_654.json')['metrics']
                self.assertEqual(len(test['predictions']), 4)
                self.assertEqual(len({r['video'] for r in test['predictions']}), 4)
                model = build_model(config['recipes']['supcon_stage2'], (3,26,6), 'head', 2)
                self.assertTrue(all(n.startswith(('classifier.', 'multi_head_attention_pooling.'))
                                    for n,p in model.named_parameters() if p.requires_grad))

    def test_incomplete_seeds_never_prepare_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config, manifest = self.setup_runs(root, 'none')
            write_json(root/'trained_models.json', dict(protocol_hash='protocol',
                        methods={'baseline': {}, 'supcon': {}}))
            with patch('stamp.pennaction.workflow.prepare') as forbidden:
                with self.assertRaisesRegex(ValueError, 'five seeds'):
                    evaluate_test(config, manifest, root, 'protocol', 'none', 'cpu')
                forbidden.assert_not_called()

    def test_interrupted_training_resumes_identical_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            config, manifest=self.setup_runs(root, 'none')
            task=dict(root=str(root),run_dir=str(root/'reference'),config=config,manifest=manifest,
                      protocol_hash='protocol',reduction='none',stage='baseline',seed=654)
            reference=fit_task(task, 'cpu')
            task['run_dir']=str(root/'interrupted')
            def save_then_stop(payload, path):
                atomic_torch_save(payload,path)
                if path.name == 'resume.pt' and payload['epoch'] == 1:
                    raise KeyboardInterrupt()
            with patch('stamp.pennaction.training.atomic_torch_save', side_effect=save_then_stop):
                with self.assertRaises(KeyboardInterrupt):
                    fit_task(task,'cpu')
            resumed=fit_task(task,'cpu')
            self.assertEqual(reference['curve'],resumed['curve'])
            expected=torch.load(root/reference['checkpoint'], weights_only=True)['model']
            actual=torch.load(root/resumed['checkpoint'], weights_only=True)['model']
            self.assertTrue(all(torch.equal(expected[k],actual[k]) for k in expected))

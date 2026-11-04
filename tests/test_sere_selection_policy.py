import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sere_selection_policy import apply_selection_policy
from stamp.sere.data import read_json, write_json
from stamp.sere.workflow import lock_artifact, select, tasks_for


class SelectionPolicyTests(unittest.TestCase):
    def setup_policy(self, root):
        config = dict(window_lengths=[16, 32, 64, 80, 96], seeds=[654, 114],
                      cv=dict(n_folds=2, seed=42), reduction='none',
                      recipes={k: {} for k in ('baseline', 'supcon_stage1', 'supcon_stage2')})
        path = root / 'config.json'
        write_json(path.with_suffix('.selection.json'), dict(window_lengths=[32, 64, 80]))
        return config, path, root / 'runs'

    def test_restriction_preserves_run_tasks_and_schedules_only_three_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, path, output = self.setup_policy(Path(tmp))
            restricted = apply_selection_policy(config, path, output, 'original-protocol')
            self.assertEqual(config['window_lengths'], [16, 32, 64, 80, 96])
            self.assertEqual(restricted['window_lengths'], [32, 64, 80])
            manifest = dict(patients=[1, 2, 3, 4])
            with patch('stamp.sere.workflow.development_path', return_value=Path('/unchanged/data')):
                args = (manifest, output, 'original-protocol', 'baseline', 32, 'baseline', {}, output / 'cv/L32')
                self.assertEqual(tasks_for(config, *args), tasks_for(restricted, *args))
                seen = []
                def fake_dispatch(tasks, devices):
                    seen.extend(t['identity']['length'] for t in tasks)
                    return {t['tag']: dict(
                        checkpoint='/unchanged/checkpoint', signature='same', epochs=1,
                        curve=[dict(epoch=1, patients={str(pid): dict(ba=.7, loss=.2)
                               for pid in t['identity']['held_out_patients']})]) for t in tasks}
                with patch('stamp.sere.workflow.dispatch', side_effect=fake_dispatch):
                    selected = select(restricted, manifest, output, 'original-protocol', 'cpu')
            self.assertEqual(set(seen), {32, 64, 80})
            for method in ('baseline', 'supcon'):
                self.assertEqual(set(selected['candidates'][method]), {'32', '64', '80'})
                self.assertEqual(selected['methods'][method]['length'], 32)
            self.assertEqual(apply_selection_policy(config, path, output, 'original-protocol'), restricted)
            # Scratch's recorded policy still applies if the sidecar is omitted on another host.
            path.with_suffix('.selection.json').unlink()
            self.assertEqual(apply_selection_policy(config, path, output, 'original-protocol'), restricted)

    def test_policy_rejects_invalid_subsets_and_later_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, path, output = self.setup_policy(Path(tmp))
            for invalid in [[], [32, 32], [128], [True], ['32']]:
                write_json(path.with_suffix('.selection.json'), dict(window_lengths=invalid))
                with self.assertRaises(ValueError):
                    apply_selection_policy(config, path, output, 'original-protocol')
            write_json(path.with_suffix('.selection.json'), dict(window_lengths=[32, 64, 80]))
            apply_selection_policy(config, path, output, 'original-protocol')
            write_json(path.with_suffix('.selection.json'), dict(window_lengths=[32, 64]))
            with self.assertRaisesRegex(ValueError, 'differs from the saved policy'):
                apply_selection_policy(config, path, output, 'original-protocol')

    def test_existing_selection_or_final_test_work_cannot_be_silently_reinterpreted(self):
        for artifact in ['final', 'test', 'data/L32/test', 'selection.json']:
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory() as tmp:
                config, path, output = self.setup_policy(Path(tmp))
                if artifact == 'selection.json':
                    lock_artifact(output / artifact, dict(protocol_hash='original-protocol',
                        candidates={m: {'16': {}} for m in ['baseline', 'supcon']},
                        methods={m: {'length': 16} for m in ['baseline', 'supcon']}))
                else:
                    (output / artifact).mkdir(parents=True)
                with self.assertRaises(ValueError):
                    apply_selection_policy(config, path, output, 'original-protocol')
                self.assertFalse((output / 'selection_policy.json').exists())


if __name__ == '__main__':
    unittest.main()

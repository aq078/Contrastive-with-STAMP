"""Shell orchestration checks using fake srun/Python; no Slurm jobs or GPUs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class BatchTests(unittest.TestCase):
    def run_pipeline(self, failure='', missing_hostname=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python = root / 'python'
            python.write_text('''#!/usr/bin/env bash
set -eu
if [[ "$1" == - ]]; then cat >/dev/null; exit 0; fi
phase="$3/$7"
printf '%s\\n' "$phase" >> "$SERE_TEST_LOG"
[[ "$phase" != "$SERE_TEST_FAILURE" ]]
''')
            srun = root / 'srun'
            srun.write_text('''#!/usr/bin/env bash
set -eu
while [[ "$1" == --* ]]; do shift; done
exec "$@"
''')
            python.chmod(0o755)
            srun.chmod(0o755)
            if missing_hostname:
                hostname = root / 'hostname'
                hostname.write_text('#!/bin/bash\necho hostname-unavailable >&2\nexit 127\n')
                hostname.chmod(0o755)
            repo = Path(__file__).resolve().parents[1]
            env = dict(os.environ, PATH=str(root) + os.pathsep + os.environ['PATH'],
                       SERE_PYTHON=str(python), SERE_REPO_DIR=str(repo),
                       SERE_TEST_LOG=str(root / 'calls'), SERE_TEST_FAILURE=failure,
                       SERE_CONFIG='configs/sere_temporal_lopo.json', SLURM_JOB_ID='mock',
                       SLURMD_NODENAME='allocated-gpu-node')
            result = subprocess.run(['bash', str(repo / 'scripts/sere_pipeline.sbatch')],
                                    env=env, capture_output=True, text=True)
            return result, (root / 'calls').read_text().splitlines()

    def test_phases_run_in_requested_order(self):
        result, calls = self.run_pipeline()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [f'{phase}/{mode}' for mode in ['none', 'mean']
                                for phase in ['prepare', 'select', 'fit', 'test']])

    def test_failure_prevents_later_training_and_test_phases(self):
        result, calls = self.run_pipeline('select/none')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(calls, ['prepare/none', 'select/none'])

    def test_missing_hostname_does_not_emit_false_failure_or_stop_job(self):
        result, calls = self.run_pipeline(missing_hostname=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('on allocated-gpu-node;', result.stdout)
        self.assertNotIn('hostname-unavailable', result.stderr)
        self.assertNotIn('Command error', result.stderr)
        self.assertEqual(len(calls), 8)

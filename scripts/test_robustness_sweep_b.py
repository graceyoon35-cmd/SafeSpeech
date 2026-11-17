"""Sweep reproducibility/index tests; protected audio here is a synthetic fixture."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile
import soundfile as sf

from run_robustness_b import ROOT
from run_robustness_sweep_b import default_conditions, run_sweep


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        audio, rate = sf.read(ROOT / 'data/LibriTTS/5339/5339_14134_000092_000003.wav')
        wavfile.write(self.directory / 'original.wav', rate, audio)
        wavfile.write(self.directory / 'fixture.wav', rate, audio * .99)
        self.manifest = {'schema_version': 1, 'experiment_id': 'synthetic_pair_fixture',
                         'audio': {'original': 'original.wav', 'protected': 'fixture.wav'},
                         'conditions': [{'name': 'gaussian_noise', 'seed': 42, 'std': .005},
                                        {'name': 'resample', 'target_hz': 8000}]}
        self.path = self.directory / 'input.json'
        self.path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_grid_has_fifteen_unique_conditions(self):
        conditions = default_conditions()
        self.assertEqual(len(conditions), 15)
        self.assertEqual(len({json.dumps(c, sort_keys=True) for c in conditions}), 15)

    def test_snapshot_csv_counts_and_reproducibility(self):
        path, summary = run_sweep(self.path, self.directory / 'runs')
        self.assertEqual(summary['status'], 'PASS')
        self.assertEqual(summary['wav_count'], 8)
        self.assertEqual(summary['mp3_count'], 0)
        self.assertTrue(summary['input_files_unchanged'])
        self.assertTrue(summary['reproducibility_pass'])
        self.assertEqual(len(summary['reproducibility']), 4)
        self.assertEqual(json.loads(path.read_text()), summary)
        with (path.parent / 'artifacts.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        for row in rows:
            self.assertTrue((path.parent / row['wav_path']).is_file())
            self.assertTrue((path.parent / row['report_path']).is_file())
        np.testing.assert_equal(json.loads((path.parent / 'manifest.json').read_text()), summary['manifest'])

    def test_reject_duplicate_conditions_and_bad_repeat_count(self):
        for count in (True, 1, 0, 2.0, 11):
            with self.subTest(count=count), self.assertRaises(ValueError):
                run_sweep(self.path, self.directory / 'runs', count)
        self.manifest['conditions'].append(self.manifest['conditions'][0])
        self.path.write_text(json.dumps(self.manifest))
        with self.assertRaises(ValueError):
            run_sweep(self.path, self.directory / 'runs')
        self.assertFalse((self.directory / 'runs').exists())

    def test_failure_is_recorded(self):
        with patch('run_robustness_sweep_b.run_pair', side_effect=RuntimeError('fixture encoder failure')):
            with self.assertRaises(RuntimeError):
                run_sweep(self.path, self.directory / 'runs')
        path = next((self.directory / 'runs').glob('*/summary.json'))
        report = json.loads(path.read_text())
        self.assertEqual(report['status'], 'FAIL')
        self.assertIn('fixture encoder failure', report['error'])


if __name__ == '__main__':
    unittest.main()

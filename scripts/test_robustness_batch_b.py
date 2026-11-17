"""Check batch indexing, pair identity and failure reporting."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from run_robustness_batch_b import run_batch


class BatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.output = self.root / 'results'
        self.rate = 16000
        time = np.arange(self.rate) / self.rate
        base = .2 * np.sin(2 * np.pi * 440 * time)
        for name, signal in [('original', base), ('spec1', base * .99), ('spec10', base * .9)]:
            sf.write(self.root / f'{name}.wav', signal, self.rate)
        self.paths = [self.manifest('spec1'), self.manifest('spec10')]

    def manifest(self, name, *, protected=None, experiment_id=None, target=8000):
        path = self.root / f'{name}.json'
        path.write_text(json.dumps({
            'schema_version': 1, 'experiment_id': experiment_id or name,
            'audio': {'original': 'original.wav', 'protected': protected or f'{name}.wav'},
            'conditions': [{'name': 'resample', 'target_hz': target}],
        }), encoding='utf-8')
        return path

    def test_same_original_two_protection_variants_build_portable_index(self):
        summary, report = run_batch(self.paths, self.output)
        self.assertEqual(report['status'], 'PASS')
        self.assertEqual(report['input_pair_count'], 2)
        self.assertEqual(report['distinct_original_sha256_count'], 1)
        self.assertEqual(report['wav_count'], 8)
        with (summary.parent / 'artifacts.csv').open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 8)
        self.assertEqual({row['experiment_id'] for row in rows}, {'spec1', 'spec10'})
        for row in rows:
            for key in ('wav_path', 'report_path'):
                self.assertFalse(Path(row[key]).is_absolute())
                self.assertTrue((summary.parent / row[key]).is_file())

    def test_invalid_batch_has_no_outputs(self):
        for path in (
            self.manifest('duplicate_id', protected='spec10.wav', experiment_id='spec1'),
            self.manifest('duplicate_pair', protected='spec1.wav'),
            self.manifest('different_conditions', protected='spec10.wav', target=16000),
        ):
            with self.subTest(path=path), self.assertRaises(ValueError):
                run_batch([self.paths[0], path], self.output)
            self.assertFalse(self.output.exists())

    def test_later_pair_failure_retains_a_fail_summary(self):
        sf.write(self.root / 'spec10.wav', np.ones(self.rate - 1) * .1, self.rate)
        with self.assertRaises(ValueError):
            run_batch(self.paths, self.output)
        summary = next(self.output.glob('*/summary.json'))
        report = json.loads(summary.read_text(encoding='utf-8'))
        self.assertEqual(report['status'], 'FAIL')
        self.assertEqual(report['input_pair_count'], 2)
        self.assertEqual(len(report['pairs']), 1)
        self.assertEqual(report['wav_count'], 4)


if __name__ == '__main__':
    unittest.main()

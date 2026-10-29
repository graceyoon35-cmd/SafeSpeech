"""Negative contract tests; fixtures here are not SPEC experiment outputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from audio_metrics import score_pair
from evaluate_spec_stoi import evaluate_pair, main, sha256


class ContractTests(unittest.TestCase):
    def test_output_failure_is_reported_as_fail(self):
        with TemporaryDirectory() as temp:
            args = ['evaluate_spec_stoi.py', '--manifests', 'unused.json',
                    '--validations', 'unused.json', '--output-root', temp]
            with (patch('sys.argv', args), patch('evaluate_spec_stoi.evaluate_pair',
                    return_value=dict(iterations=1, original_sha256='fixture')),
                    patch('evaluate_spec_stoi.csv.DictWriter', side_effect=OSError('simulated storage failure')),
                    patch('builtins.print')):
                self.assertEqual(main(), 1)
            reports = list(Path(temp).glob('spec_stoi_*/report.json'))
            self.assertEqual(len(reports), 1)
            report = json.loads(reports[0].read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertEqual(report['error_type'], 'OSError')

    def test_different_originals_cannot_be_compared(self):
        with TemporaryDirectory() as temp:
            args = ['evaluate_spec_stoi.py', '--manifests', 'a.json', 'b.json',
                    '--validations', 'a.json', 'b.json', '--output-root', temp]
            pairs = [dict(iterations=1, original_sha256='fixture-a'),
                     dict(iterations=10, original_sha256='fixture-b')]
            with (patch('sys.argv', args), patch('evaluate_spec_stoi.evaluate_pair', side_effect=pairs),
                    patch('builtins.print')):
                self.assertEqual(main(), 1)
            report = json.loads(next(Path(temp).glob('spec_stoi_*/report.json')).read_text())
            self.assertEqual(report['status'], 'FAIL')
            self.assertIn('same original WAV', report['error'])

    def test_different_rates_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identical sample rate'):
            score_pair(np.ones(24000), np.ones(24000), 24000, 16000)

    def test_different_lengths_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'identical sample rate and length'):
            score_pair(np.ones(24000), np.ones(23999), 24000, 24000)

    def test_wrong_roles_are_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'validation.json').write_text('{}')
            (folder / 'manifest.json').write_text(json.dumps(dict(schema_version=1,
                evaluation=dict(reference_role='original', target_role='synthetic', alignment='strict'))))
            with self.assertRaisesRegex(ValueError, 'strict original'):
                evaluate_pair(folder / 'manifest.json', folder / 'validation.json')

    def test_synthetic_role_is_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / 'validation.json').write_text('{}')
            (folder / 'manifest.json').write_text(json.dumps(dict(schema_version=1,
                evaluation=dict(reference_role='original', target_role='protected', alignment='strict'),
                audio=dict(original='clean.wav', protected='protected.wav', synthetic='mock.wav'))))
            with self.assertRaisesRegex(ValueError, 'synthetic=null'):
                evaluate_pair(folder / 'manifest.json', folder / 'validation.json')

    def test_artifact_changed_after_validation_is_rejected(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp).resolve()
            for name in ('clean.wav', 'protected.wav', 'perturbation.pt'):
                (folder / name).write_bytes(b'contract fixture, not audio')
            (folder / 'report.json').write_text(json.dumps(dict(status='PASS', mode='SPEC', iterations=1)))
            hashes = {str(folder / name): sha256(folder / name)
                      for name in ('clean.wav', 'protected.wav', 'perturbation.pt', 'report.json')}
            (folder / 'validation.json').write_text(json.dumps(dict(status='PASS', run_dir=str(folder),
                iterations=1, source_copy_identical=True, saved_noise_reconstructs_wav=True,
                files_sha256=hashes)))
            (folder / 'manifest.json').write_text(json.dumps(dict(schema_version=1,
                evaluation=dict(reference_role='original', target_role='protected', alignment='strict'),
                audio=dict(original='clean.wav', protected='protected.wav', synthetic=None))))
            (folder / 'protected.wav').write_bytes(b'changed fixture')
            with self.assertRaisesRegex(ValueError, 'changed since validation'):
                evaluate_pair(folder / 'manifest.json', folder / 'validation.json')


if __name__ == '__main__':
    unittest.main()
